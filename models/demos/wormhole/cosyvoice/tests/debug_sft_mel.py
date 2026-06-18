#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-SFT mel-level investigation of the "quiet wav" symptom.

Uses the bit-exact-verified sft_en golden (real prompts + real embedding +
53 speech tokens -> mel [1,80,91]). Feeds the SAME golden speech tokens to
BOTH the reference flow and the TT flow (matched noise), so the comparison
isolates the FLOW from LLM divergence.

Reports:
  A. End-to-end mel PCC + magnitude (RMS/std/min/max) for ref vs TT.
  B. Per-stage: encoder_out, encoder_proj, regulator(mu), spks, cond, mask.
  C. Per-Euler-step mel PCC + magnitude drift (does TT shrink or grow?).
  D. The same with EMPTY prompts (compare.py regime) to test whether the
     magnitude collapse is specific to the empty-prompt SFT path.

Matched-noise: construct TT flow BEFORE seeding (per HANDOFF §11).
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = os.getenv(
    "COSYVOICE_MODEL_DIR",
    str(PROJECT_ROOT.parent.parent.parent.parent) + "/pretrained_models/CosyVoice-300M",
)
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice"))
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice" / "third_party" / "Matcha-TTS"))

from cosyvoice.utils.mask import make_pad_mask

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_flow import TtCosyVoiceFlow


def stats(t, name):
    t = t.float()
    rms = t.pow(2).mean().sqrt().item()
    print(
        f"  {name}: shape={tuple(t.shape)} min={t.min():.4f} max={t.max():.4f} "
        f"mean={t.mean():.4f} std={t.std():.4f} rms={rms:.4f}"
    )


def comp(ref, out, name):
    if ref.shape != out.shape:
        print(f"  [{name}] SHAPE MISMATCH ref={tuple(ref.shape)} out={tuple(out.shape)}")
        return float("nan")
    _, pcc = comp_pcc(ref, out, pcc=0.0)
    diff = (ref - out).abs()
    print(
        f"  [{name}] PCC={pcc:.6f} max_diff={diff.max():.6f} mean_diff={diff.mean():.6f} "
        f"ref_rms={ref.pow(2).mean().sqrt():.4f} out_rms={out.pow(2).mean().sqrt():.4f}"
    )
    return pcc


def ref_solver_steps(ref_cfm, mu, mask, spks, cond, n_timesteps=10, seed=1234):
    ref_cfm.inference_cfg_rate = 0.0
    torch.manual_seed(seed)
    z = torch.randn_like(mu).to(mu.device).to(mu.dtype)
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t.unsqueeze(0)
    x = z.clone()
    xs = [x.clone()]
    with torch.no_grad():
        for step in range(1, len(t_span)):
            dphi_dt = ref_cfm.estimator(x, mask, mu, t, spks, cond, streaming=False)
            x = x + dt * dphi_dt
            t = t + dt
            xs.append(x.clone())
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
    return xs


def tt_solver_steps(tt_cfm, device, mu, mask, spks, cond, n_timesteps=10, seed=1234):
    from models.demos.wormhole.cosyvoice.tt.cosyvoice_flow import _ensure_4d_mask, _ensure_spks_4d

    tt_cfm.inference_cfg_rate = 0.0
    T = mu.shape[-1]
    B = mu.shape[0]
    mu_t = mu.float()
    torch.manual_seed(seed)
    z_t = torch.randn_like(mu_t)  # [B,80,T]
    z_t = z_t.permute(0, 2, 1).contiguous().reshape(B, 1, T, 80)
    z_tt = ttnn.from_torch(z_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    mu_tt = ttnn.from_torch(mu.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mask_4d = _ensure_4d_mask(ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
    spks_4d = _ensure_spks_4d(ttnn.from_torch(spks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device), T)
    cond_4d = ttnn.from_torch(
        cond.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    t_span = torch.linspace(0, 1, n_timesteps + 1, device="cpu", dtype=torch.float32)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

    x = z_tt
    xs = [ttnn.to_torch(x).float()]
    with torch.no_grad():
        for step in range(1, n_timesteps + 1):
            t_cur = t_span[step - 1].unsqueeze(0)
            dphi = tt_cfm.estimator.forward(x=x, mask=mask_4d, mu=mu_tt, t=t_cur, spks=spks_4d, cond=cond_4d)
            dt = (t_span[step] - t_span[step - 1]).item()
            dphi_scaled = ttnn.multiply(dphi, dt)
            x = ttnn.add(x, dphi_scaled)
            xs.append(ttnn.to_torch(x).float())
    return xs


def to_80_T(x_tt_4d):
    return x_tt_4d.squeeze(1).transpose(1, 2).contiguous()


def extract_decoder_inputs(flow, inputs, is_tt, device):
    """Reproduce flow.inference pre-decoder staging. Returns dict with
    mu [B,80,T], mask [B,1,T], spks [B,80], cond [B,80,T], mel_len1, mel_len2."""
    with torch.no_grad():
        token = inputs["token"]
        prompt_token = inputs["prompt_token"]
        prompt_feat = inputs["prompt_feat"]
        embedding = inputs["embedding"]
        token_len = inputs["token_len"]
        prompt_token_len = inputs["prompt_token_len"]

        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token_all = torch.cat([prompt_token, token], dim=1)
        token_len_all = prompt_token_len + token_len

        if is_tt:
            mask = (~flow._make_pad_mask(token_len_all)).unsqueeze(-1).to(embedding)
            token_emb = flow.input_embedding(torch.clamp(token_all, min=0)) * mask
            token_tt = ttnn.from_torch(token_emb, layout=ttnn.TILE_LAYOUT, device=device)
            h_tt, _ = flow.encoder(token_tt, token_len=token_len_all)
            h = ttnn.to_torch(h_tt).float()
            h = flow.encoder_proj(h)
            mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / 50 * 22050 / 256)
            h_reg_tt, _ = flow.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, 50)
            h_reg = ttnn.to_torch(h_reg_tt).float() if isinstance(h_reg_tt, ttnn.Tensor) else h_reg_tt.float()
            mask_cpu = (~flow._make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).float()
            embedding_norm = F.normalize(embedding, dim=1)
            spk = flow.spk_embed_affine_layer(embedding_norm)
        else:
            embedding = F.normalize(embedding, dim=1)
            spk = flow.spk_embed_affine_layer(embedding)
            mask = (~make_pad_mask(token_len_all)).unsqueeze(-1).to(spk)
            token_emb = flow.input_embedding(torch.clamp(token_all, min=0)) * mask
            h, _ = flow.encoder(token_emb, token_len_all)
            h = flow.encoder_proj(h)
            mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / flow.input_frame_rate * 22050 / 256)
            h, _ = flow.length_regulator.inference(
                h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, flow.input_frame_rate
            )
            h_reg = h
            mask_cpu = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h_reg)

        conds = torch.zeros([1, mel_len1 + mel_len2, 80], device=embedding.device, dtype=torch.float32)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)
        mu = h_reg.transpose(1, 2).contiguous()
        return dict(mu=mu, mask=mask_cpu.unsqueeze(1), spks=spk, cond=conds, mel_len1=mel_len1, mel_len2=mel_len2)


def run_case(label, ref_flow, tt_flow, device, inputs, do_steps=True):
    print(f"\n{'='*60}\n=== CASE: {label}  (T_tok={inputs['token'].shape[1]})\n{'='*60}")
    # Full flow E2E (matched noise)
    original_cfg = ref_flow.decoder.inference_cfg_rate
    ref_flow.decoder.inference_cfg_rate = 0.0
    torch.manual_seed(1234)
    with torch.no_grad():
        ref_feat, _ = ref_flow.inference(**inputs)
    ref_flow.decoder.inference_cfg_rate = original_cfg
    torch.manual_seed(1234)
    with torch.no_grad():
        tt_feat, _ = tt_flow.inference(**inputs)
    tt_feat = tt_feat.squeeze(1).transpose(1, 2).contiguous()
    stats(ref_feat, f"{label} ref_feat")
    stats(tt_feat, f"{label} tt_feat")
    comp(ref_feat, tt_feat, f"{label} E2E_mel")

    # Per-stage
    print(f"--- {label} per-stage ---")
    ref_di = extract_decoder_inputs(ref_flow, inputs, is_tt=False, device=device)
    tt_di = extract_decoder_inputs(tt_flow, inputs, is_tt=True, device=device)
    comp(ref_di["mu"], tt_di["mu"], f"{label} mu")
    stats(ref_di["mu"], f"{label} ref_mu")
    stats(tt_di["mu"], f"{label} tt_mu")
    comp(ref_di["spks"], tt_di["spks"], f"{label} spks")
    comp(ref_di["cond"], tt_di["cond"], f"{label} cond")
    comp(ref_di["mask"].float(), tt_di["mask"].float(), f"{label} mask")

    # Per-Euler-step (using ref mu for both, matched noise)
    if do_steps:
        print(f"--- {label} per-Euler-step (ref_mu -> both) ---")
        ref_xs = ref_solver_steps(
            ref_flow.decoder, ref_di["mu"], ref_di["mask"], ref_di["spks"], ref_di["cond"], seed=1234
        )
        tt_xs = tt_solver_steps(
            tt_flow.decoder, device, ref_di["mu"], ref_di["mask"], ref_di["spks"], ref_di["cond"], seed=1234
        )
        print("step | PCC | ref_rms | tt_rms")
        for i, (rx, tx) in enumerate(zip(ref_xs, tt_xs)):
            tx_80 = to_80_T(tx)
            _, pcc = comp_pcc(rx, tx_80, pcc=0.0)
            print(f"  {i:3d} | {pcc:.6f} | {rx.pow(2).mean().sqrt():.4f} | {tx_80.pow(2).mean().sqrt():.4f}")


def main():
    print("=== Loading reference model ===")
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_flow = ref_model.flow
    ref_flow.eval()

    device = ttnn.open_device(device_id=0, l1_small_size=64 << 10, trace_region_size=128 << 20)
    device.enable_program_cache()
    tt_flow = TtCosyVoiceFlow(device, state_dict=ref_flow.state_dict(), ref_flow=ref_flow)

    # ---- CASE 1: golden sft_en (real prompts, 53 tokens) ----
    print("\nLoading golden sft_en ...")
    golden_inputs = torch.load(GOLDEN_DIR / "inputs" / "sft_en.pt", map_location="cpu", weights_only=False)
    golden_tokens = torch.load(GOLDEN_DIR / "tokens" / "sft_en.pt", map_location="cpu", weights_only=False)
    golden_mel = torch.load(GOLDEN_DIR / "mels" / "sft_en.pt", map_location="cpu", weights_only=False)
    print(f"golden tokens shape={tuple(golden_tokens.shape)} golden mel shape={tuple(golden_mel.shape)}")

    # Build flow.inference inputs from the golden: token = golden speech tokens,
    # prompt = golden flow_prompt_speech_token + prompt_speech_feat.
    flow_inputs = {
        "token": golden_tokens.unsqueeze(0).to(torch.int32),
        "token_len": torch.tensor([golden_tokens.shape[0]], dtype=torch.int32),
        "prompt_token": golden_inputs["flow_prompt_speech_token"],
        "prompt_token_len": golden_inputs["flow_prompt_speech_token_len"],
        "prompt_feat": golden_inputs["prompt_speech_feat"],
        "prompt_feat_len": golden_inputs["prompt_speech_feat_len"],
        "embedding": golden_inputs["flow_embedding"],
        "flow_cache": torch.zeros(1, 80, 0, 2, dtype=torch.float32),
    }
    run_case("golden_sft_en", ref_flow, tt_flow, device, flow_inputs, do_steps=True)

    # Sanity: reference flow mel should match the golden mel (bit-exact-ish)
    original_cfg = ref_flow.decoder.inference_cfg_rate
    ref_flow.decoder.inference_cfg_rate = 0.0
    torch.manual_seed(0)  # golden was generated with sampling_seed=0; flow noise uses global RNG
    with torch.no_grad():
        ref_check, _ = ref_flow.inference(**flow_inputs)
    ref_flow.decoder.inference_cfg_rate = original_cfg
    print(f"\n[golden sanity] ref_flow mel (seed=0) vs saved golden mel:")
    comp(golden_mel, ref_check, "golden_vs_ref_seed0")

    # ---- CASE 2: empty-prompt SFT (compare.py regime) ----
    # Use a short text + the en_speaker_6s embedding, but EMPTY prompts.
    print("\n=== Building empty-prompt SFT inputs (compare.py regime) ===")
    from models.demos.wormhole.cosyvoice.reference.golden_pipeline import _encode_text, _get_predefined_speakers

    spk = _get_predefined_speakers()["en_speaker_6s"]
    emb = spk["llm_embedding"].clone()
    text_tok = _encode_text("Hello world, this is a test of English synthesis.")
    # Need speech tokens: use the golden sft_en tokens as a stand-in token stream
    # (the point is to test the FLOW, not the LLM). 53 tokens -> ~2s mel.
    empty_inputs = {
        "token": golden_tokens.unsqueeze(0).to(torch.int32),
        "token_len": torch.tensor([golden_tokens.shape[0]], dtype=torch.int32),
        "prompt_token": torch.zeros(1, 0, dtype=torch.int32),
        "prompt_token_len": torch.tensor([0], dtype=torch.int32),
        "prompt_feat": torch.zeros(1, 0, 80, dtype=torch.float32),
        "prompt_feat_len": torch.tensor([0], dtype=torch.int32),
        "embedding": emb,
        "flow_cache": torch.zeros(1, 80, 0, 2, dtype=torch.float32),
    }
    run_case("empty_prompt_sft", ref_flow, tt_flow, device, empty_inputs, do_steps=True)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
