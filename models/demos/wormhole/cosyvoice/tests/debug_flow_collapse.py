#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Flow magnitude-collapse diagnostic.

Reproduces the 0.16 full-flow PCC, then isolates the cause via four experiments:

  A. Baseline: full ref flow vs full TT flow, matched noise (expect ~0.16).
  B. Real-mu cross-feed: extract REAL mu from each path's regulator, then feed
     the SAME real mu to BOTH solvers with matched noise. If PCC jumps to ~0.65
     (the random-mu isolated number), the full-flow gap is NOT the mu — it's
     mask/cond/spks layout or padding. If it stays ~0.16, real-mu magnitude is
     the cause.
  C. mu magnitude: report std/min/max of ref_mu and tt_mu.
  D. Per-step mel: dump x at each Euler step for both ref and TT (using real mu)
     to see whether divergence appears at step 1 or accumulates.

Matched-noise rule (per HANDOFF §11): construct the TT flow BEFORE seeding, so
the TT CFM's `z = torch.randn_like(mu)` draws from the same RNG state as the
reference's `z`.
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
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice"))
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice" / "third_party" / "Matcha-TTS"))

from cosyvoice.utils.mask import make_pad_mask

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_flow import TtCosyVoiceFlow


def stats(t, name):
    t = t.float()
    print(
        f"  {name}: shape={tuple(t.shape)} min={t.min():.4f} max={t.max():.4f} "
        f"mean={t.mean():.4f} std={t.std():.4f} abs_mean={t.abs().mean():.4f}"
    )


def comp(ref, out, name):
    if ref.shape != out.shape:
        print(f"  [{name}] SHAPE MISMATCH ref={tuple(ref.shape)} out={tuple(out.shape)}")
        return float("nan")
    _, pcc = comp_pcc(ref, out, pcc=0.0)
    diff = (ref - out).abs()
    print(
        f"  [{name}] PCC={pcc:.6f} max_diff={diff.max():.6f} mean_diff={diff.mean():.6f} "
        f"ref_std={ref.std():.4f} out_std={out.std():.4f}"
    )
    return pcc


def build_inputs(ref_flow, seed=42):
    """Deterministic synthetic inputs (same shape as test_flow.py)."""
    torch.manual_seed(seed)
    B = 1
    token_len2 = 5
    token_len1 = 5
    mel_len1 = 10
    token = torch.randint(0, 4096, (B, token_len2), dtype=torch.int32)
    prompt_token = torch.randint(0, 4096, (B, token_len1), dtype=torch.int32)
    prompt_feat = torch.randn(B, mel_len1, ref_flow.output_size, dtype=torch.float32)
    embedding = torch.randn(B, 192, dtype=torch.float32)
    return dict(
        token=token,
        token_len=torch.tensor([token_len2], dtype=torch.int32),
        prompt_token=prompt_token,
        prompt_token_len=torch.tensor([token_len1], dtype=torch.int32),
        prompt_feat=prompt_feat,
        prompt_feat_len=torch.tensor([mel_len1], dtype=torch.int32),
        embedding=embedding,
        flow_cache=torch.zeros(B, 80, 0, 2, dtype=torch.float32),
    )


def run_ref_flow_full(ref_flow, inputs):
    """Run the reference flow end-to-end (CFG=0). Returns (feat, mu, mask, spks, cond)."""
    original_cfg = ref_flow.decoder.inference_cfg_rate
    ref_flow.decoder.inference_cfg_rate = 0.0
    torch.manual_seed(1234)
    with torch.no_grad():
        feat, _ = ref_flow.inference(**inputs)
    ref_flow.decoder.inference_cfg_rate = original_cfg
    return feat


def run_tt_flow_full(tt_flow, inputs):
    torch.manual_seed(1234)
    with torch.no_grad():
        feat, _ = tt_flow.inference(**inputs)
    return feat


def extract_ref_decoder_inputs(ref_flow, inputs):
    """Reproduce the reference flow.inference pre-decoder staging to get the
    exact mu/mask/spks/cond the reference decoder receives."""
    with torch.no_grad():
        token = inputs["token"]
        prompt_token = inputs["prompt_token"]
        prompt_feat = inputs["prompt_feat"]
        embedding = inputs["embedding"]
        token_len = inputs["token_len"]
        prompt_token_len = inputs["prompt_token_len"]

        embedding = F.normalize(embedding, dim=1)
        embedding = ref_flow.spk_embed_affine_layer(embedding)

        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token_all = torch.concat([prompt_token, token], dim=1)
        token_len_all = prompt_token_len + token_len
        mask = (~make_pad_mask(token_len_all)).unsqueeze(-1).to(embedding)
        token_emb = ref_flow.input_embedding(torch.clamp(token_all, min=0)) * mask

        h, _ = ref_flow.encoder(token_emb, token_len_all)
        h = ref_flow.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / ref_flow.input_frame_rate * 22050 / 256)
        h, _ = ref_flow.length_regulator.inference(
            h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, ref_flow.input_frame_rate
        )

        conds = torch.zeros([1, mel_len1 + mel_len2, ref_flow.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        mu = h.transpose(1, 2).contiguous()
        return dict(
            mu=mu,
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            mel_len1=mel_len1,
            mel_len2=mel_len2,
        )


def extract_tt_decoder_inputs(tt_flow, inputs):
    """Reproduce the TT flow.inference pre-decoder staging (CPU parts)."""
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
        mask = (~tt_flow._make_pad_mask(token_len_all)).unsqueeze(-1).to(embedding)
        token_emb = tt_flow.input_embedding(torch.clamp(token_all, min=0)) * mask

        token_tt = ttnn.from_torch(token_emb, layout=ttnn.TILE_LAYOUT, device=tt_flow.device)
        h_tt, _ = tt_flow.encoder(token_tt, token_len=token_len_all)
        h_cpu = ttnn.to_torch(h_tt).float()
        h_cpu = tt_flow.encoder_proj(h_cpu)

        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / 50 * 22050 / 256)
        h_reg_tt, _ = tt_flow.length_regulator.inference(
            h_cpu[:, :token_len1], h_cpu[:, token_len1:], mel_len1, mel_len2, 50
        )
        h_reg = ttnn.to_torch(h_reg_tt).float() if isinstance(h_reg_tt, ttnn.Tensor) else h_reg_tt.float()

        conds = torch.zeros([1, mel_len1 + mel_len2, 80], device=embedding.device, dtype=torch.float32)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask_cpu = (~tt_flow._make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).float()
        embedding_norm = F.normalize(embedding, dim=1)
        embedding = tt_flow.spk_embed_affine_layer(embedding_norm)

        mu = h_reg.transpose(1, 2).contiguous()
        return dict(
            mu=mu,
            mask=mask_cpu.unsqueeze(1),
            spks=embedding,
            cond=conds,
            mel_len1=mel_len1,
            mel_len2=mel_len2,
        )


def ref_solver_with_steps(ref_cfm, mu, mask, spks, cond, n_timesteps=10, seed=1234):
    """Reimplement the reference solve_euler so we can capture per-step x.
    CFG=0 (single batch). Returns list of x per step (incl. initial z as step 0)."""
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
            # Reference forward_estimator expects the 2x-batch layout but with
            # CFG=0 the uncond half is zeroed; the CFG combination reduces to
            # just the cond half. Simpler: call estimator directly on B=1.
            dphi_dt = ref_cfm.estimator(x, mask, mu, t, spks, cond, streaming=False)
            # CFG=0: dphi = (1+0)*cond - 0*uncond = cond. Direct call already
            # gives the conditioned output, so no CFG math needed.
            x = x + dt * dphi_dt
            t = t + dt
            xs.append(x.clone())
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
    return xs


def tt_solver_with_steps(tt_cfm, device, mu, mask, spks, cond, n_timesteps=10, seed=1234):
    """Reimplement the TT solve_euler so we can capture per-step x.
    CFG=0 (single batch). Returns list of x per step (as torch [B,80,T])."""
    tt_cfm.inference_cfg_rate = 0.0
    from models.demos.wormhole.cosyvoice.tt.cosyvoice_flow import _ensure_4d_mask, _ensure_spks_4d

    T = mu.shape[-1]
    B = mu.shape[0]
    mu_t = mu.float()
    torch.manual_seed(seed)
    z_t = torch.randn_like(mu_t)  # [B,80,T]
    # TT estimator wants x in [B,1,T,80]. z is [B,80,T] -> permute -> [B,T,80] -> reshape [B,1,T,80].
    z_t = z_t.permute(0, 2, 1).contiguous()  # [B,T,80]
    z_t = z_t.reshape(B, 1, T, 80)  # [B,1,T,80]
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
            dphi = tt_cfm.estimator.forward(
                x=x,
                mask=mask_4d,
                mu=mu_tt,
                t=t_cur,
                spks=spks_4d,
                cond=cond_4d,
            )
            dt = (t_span[step] - t_span[step - 1]).item()
            dphi_scaled = ttnn.multiply(dphi, dt)
            x = ttnn.add(x, dphi_scaled)
            xs.append(ttnn.to_torch(x).float())
    return xs


def to_80_T(x_tt_4d):
    """TT x is [B,1,T,80] -> [B,80,T] to match reference layout."""
    return x_tt_4d.squeeze(1).transpose(1, 2).contiguous()


def main():
    print("=== Loading reference model ===")
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_flow = ref_model.flow
    ref_flow.eval()

    device = ttnn.open_device(device_id=0, l1_small_size=64 << 10, trace_region_size=128 << 20)
    device.enable_program_cache()

    inputs = build_inputs(ref_flow, seed=42)

    # ---- Experiment A0: construct tt_flow AFTER ref run (matches test_flow.py order) ----
    print("\n=== Experiment A0: tt_flow constructed AFTER ref run (test_flow.py order) ===")
    ref_feat_a0 = run_ref_flow_full(ref_flow, inputs)
    tt_flow_a0 = TtCosyVoiceFlow(device, state_dict=ref_flow.state_dict(), ref_flow=ref_flow)
    tt_feat_a0 = run_tt_flow_full(tt_flow_a0, inputs)
    tt_feat_a0 = tt_feat_a0.squeeze(1).transpose(1, 2).contiguous()
    comp(ref_feat_a0, tt_feat_a0, "A0.full_flow_construct_after")

    print("=== Constructing TT flow (canonical, BEFORE seeding, per matched-noise rule) ===")
    tt_flow = TtCosyVoiceFlow(device, state_dict=ref_flow.state_dict(), ref_flow=ref_flow)

    # ---- Experiment A: full flow baseline (input-seed sweep) ----
    print("\n=== Experiment A: full flow baseline (matched noise, input-seed sweep) ===")
    for iseed in [42, 0, 7, 123, 999]:
        inputs = build_inputs(ref_flow, seed=iseed)
        ref_feat = run_ref_flow_full(ref_flow, inputs)
        tt_feat = run_tt_flow_full(tt_flow, inputs)
        tt_feat = tt_feat.squeeze(1).transpose(1, 2).contiguous()
        print(f"[A seed={iseed}] ", end="")
        comp(ref_feat, tt_feat, f"full_flow_seed{iseed}")
    # Rebuild canonical inputs (seed 42) for the rest
    inputs = build_inputs(ref_flow, seed=42)
    ref_feat = run_ref_flow_full(ref_flow, inputs)
    tt_feat = run_tt_flow_full(tt_flow, inputs)
    tt_feat = tt_feat.squeeze(1).transpose(1, 2).contiguous()
    print("Ref feat:")
    stats(ref_feat, "ref_feat")
    print("TT feat:")
    stats(tt_feat, "tt_feat")

    # ---- Experiment C: mu magnitude (cheap, do before B) ----
    print("\n=== Experiment C: decoder-input magnitude ===")
    ref_di = extract_ref_decoder_inputs(ref_flow, inputs)
    tt_di = extract_tt_decoder_inputs(tt_flow, inputs)
    stats(ref_di["mu"], "ref_mu")
    stats(tt_di["mu"], "tt_mu")
    stats(ref_di["mask"].float(), "ref_mask")
    stats(tt_di["mask"].float(), "tt_mask")
    stats(ref_di["spks"], "ref_spks")
    stats(tt_di["spks"], "tt_spks")
    stats(ref_di["cond"], "ref_cond")
    stats(tt_di["cond"], "tt_cond")
    comp(ref_di["mu"], tt_di["mu"], "C.mu")
    comp(ref_di["spks"], tt_di["spks"], "C.spks")
    comp(ref_di["cond"], tt_di["cond"], "C.cond")
    comp(ref_di["mask"].float(), tt_di["mask"].float(), "C.mask")

    # ---- Experiment B: real-mu cross-feed (same real mu to BOTH solvers) ----
    print("\n=== Experiment B: real-mu cross-feed ===")
    # B1: ref_mu -> both solvers
    print("[B1] ref_mu -> ref solver  vs  ref_mu -> TT solver")
    ref_xs_b1 = ref_solver_with_steps(
        ref_flow.decoder, ref_di["mu"], ref_di["mask"], ref_di["spks"], ref_di["cond"], seed=1234
    )
    tt_xs_b1 = tt_solver_with_steps(
        tt_flow.decoder, device, ref_di["mu"], ref_di["mask"], ref_di["spks"], ref_di["cond"], seed=1234
    )
    ref_out_b1 = ref_xs_b1[-1]
    tt_out_b1 = to_80_T(tt_xs_b1[-1])
    comp(ref_out_b1, tt_out_b1, "B1.refmu_both")

    # B2: tt_mu -> both solvers
    print("[B2] tt_mu -> ref solver  vs  tt_mu -> TT solver")
    ref_xs_b2 = ref_solver_with_steps(
        ref_flow.decoder, tt_di["mu"], tt_di["mask"], tt_di["spks"], tt_di["cond"], seed=1234
    )
    tt_xs_b2 = tt_solver_with_steps(
        tt_flow.decoder, device, tt_di["mu"], tt_di["mask"], tt_di["spks"], tt_di["cond"], seed=1234
    )
    ref_out_b2 = ref_xs_b2[-1]
    tt_out_b2 = to_80_T(tt_xs_b2[-1])
    comp(ref_out_b2, tt_out_b2, "B2.ttmu_both")

    # ---- Experiment D: per-step mel divergence ----
    print("\n=== Experiment D: per-step mel PCC (ref_mu -> both, matched noise) ===")
    print("step | PCC(ref_x, tt_x) | ref_std | tt_std")
    for i, (rx, tx) in enumerate(zip(ref_xs_b1, tt_xs_b1)):
        tx_80 = to_80_T(tx)
        _, pcc = comp_pcc(rx, tx_80, pcc=0.0)
        print(f"  {i:3d} | {pcc:.6f} | {rx.std():.4f} | {tx_80.std():.4f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
