#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Quick debug: compare each stage of flow path without running the slow decoder."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = str(PROJECT_ROOT.parent.parent.parent.parent) + "/pretrained_models/CosyVoice-300M"

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_flow import TtCosyVoiceFlow

sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice"))
from cosyvoice.utils.mask import make_pad_mask


def comp(ref, out, name):
    if ref is None or out is None:
        print(f"[{name}] None tensor")
        return
    if ref.shape != out.shape:
        print(f"[{name}] SHAPE MISMATCH ref={ref.shape} out={out.shape}")
        return
    diff = (ref - out).abs()
    _, pcc = comp_pcc(ref, out, pcc=0.0)
    print(f"[{name}] PCC={pcc:.6f}  max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")


def main():
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_flow = ref_model.flow
    ref_flow.eval()

    batch_size = 1
    token_len2 = 5
    token_len1 = 5
    mel_len1 = 10

    torch.manual_seed(42)
    token = torch.randint(0, 4096, (batch_size, token_len2), dtype=torch.int32)
    prompt_token = torch.randint(0, 4096, (batch_size, token_len1), dtype=torch.int32)
    prompt_feat = torch.randn(batch_size, mel_len1, ref_flow.output_size, dtype=torch.float32)
    embedding = torch.randn(batch_size, 192, dtype=torch.float32)

    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    tt_flow = TtCosyVoiceFlow(device, state_dict=ref_flow.state_dict(), ref_flow=ref_flow)

    # ---- Stage 1: token_emb ----
    token_all = torch.cat([prompt_token, token], dim=1)
    token_len_all = torch.tensor([token_len1 + token_len2], dtype=torch.int32)
    mask = (~make_pad_mask(token_len_all)).unsqueeze(-1).to(embedding.dtype)
    token_emb = tt_flow.input_embedding(torch.clamp(token_all, min=0)) * mask

    with torch.no_grad():
        ref_mask = (~make_pad_mask(token_len_all)).unsqueeze(-1).to(ref_flow.input_embedding.weight.dtype)
        ref_token_emb = ref_flow.input_embedding(torch.clamp(token_all, min=0)) * ref_mask
    comp(ref_token_emb, token_emb, "stage1_token_emb")

    # ---- Stage 2: encoder output ----
    token_tt = ttnn.from_torch(token_emb, layout=ttnn.TILE_LAYOUT, device=device)
    h_tt, _ = tt_flow.encoder(token_tt, token_len=token_len_all)
    h_cpu = ttnn.to_torch(h_tt).float()

    with torch.no_grad():
        ref_h, _ = ref_flow.encoder(token_emb, token_len_all)
    comp(ref_h, h_cpu, "stage2_encoder_out")

    # ---- Stage 3: encoder_proj ----
    h_proj = tt_flow.encoder_proj(h_cpu)
    with torch.no_grad():
        ref_h_proj = ref_flow.encoder_proj(ref_h)
    comp(ref_h_proj, h_proj, "stage3_encoder_proj")

    # ---- Stage 4: regulator ----
    mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / 50 * 22050 / 256)
    h_reg_tt, _ = tt_flow.length_regulator.inference(
        h_proj[:, :token_len1], h_proj[:, token_len1:], mel_len1, mel_len2, 50
    )
    h_reg = ttnn.to_torch(h_reg_tt).float() if isinstance(h_reg_tt, ttnn.Tensor) else h_reg_tt.float()

    with torch.no_grad():
        ref_h_reg, _ = ref_flow.length_regulator.inference(
            ref_h_proj[:, :token_len1], ref_h_proj[:, token_len1:], mel_len1, mel_len2, 50
        )
    comp(ref_h_reg, h_reg, "stage4_regulator")

    # ---- Stage 5: decoder inputs ----
    conds = torch.zeros([1, mel_len1 + mel_len2, 80], device=embedding.device, dtype=torch.float32)
    conds[:, :mel_len1] = prompt_feat
    conds = conds.transpose(1, 2)
    mask_cpu = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).float()
    embedding_norm = F.normalize(embedding, dim=1)
    spk_embed = tt_flow.spk_embed_affine_layer(embedding_norm)

    with torch.no_grad():
        ref_embedding_norm = F.normalize(embedding, dim=1)
        ref_spk_embed = ref_flow.spk_embed_affine_layer(ref_embedding_norm)
    comp(ref_spk_embed, spk_embed, "stage5_spk_embed")

    # ---- Stage 6: mu (before decoder) ----
    mu = h_reg.transpose(1, 2).contiguous()
    ref_mu = ref_h_reg.transpose(1, 2).contiguous()
    comp(ref_mu, mu, "stage6_mu")

    # ---- Stage 7: full decoder with 1 timestep for speed ----
    print("\n--- Running decoder (1 timestep for speed) ---")
    tt_feat_tt, _ = tt_flow.decoder(
        mu=ttnn.from_torch(mu, layout=ttnn.TILE_LAYOUT, device=device),
        mask=ttnn.from_torch(mask_cpu.unsqueeze(1), layout=ttnn.TILE_LAYOUT, device=device),
        n_timesteps=1,
        spks=ttnn.from_torch(spk_embed, layout=ttnn.TILE_LAYOUT, device=device),
        cond=ttnn.from_torch(conds, layout=ttnn.TILE_LAYOUT, device=device),
        prompt_len=mel_len1,
        cache=torch.zeros(batch_size, 80, 0, 2, dtype=torch.float32),
    )

    with torch.no_grad():
        ref_feat, _ = ref_flow.decoder(
            mu=ref_mu,
            mask=mask_cpu.unsqueeze(1),
            n_timesteps=1,
            spks=ref_spk_embed,
            cond=conds,
            prompt_len=mel_len1,
            cache=torch.zeros(batch_size, 80, 0, 2, dtype=torch.float32),
        )

    tt_feat = ttnn.to_torch(tt_feat_tt).float()
    print(f"Ref feat shape: {ref_feat.shape}, TT feat shape: {tt_feat.shape}")
    comp(ref_feat, tt_feat, "stage7_decoder_out_1step")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
