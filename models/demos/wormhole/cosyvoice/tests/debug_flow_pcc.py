#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

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


def comp_tensors(ref, out, name):
    if ref is None or out is None:
        print(f"[{name}] None tensor encountered")
        return
    if ref.shape != out.shape:
        print(f"[{name}] SHAPE MISMATCH ref={ref.shape} out={out.shape}")
        return
    diff = (ref - out).abs()
    pcc_result, pcc_val = comp_pcc(ref, out, pcc=0.0)
    print(f"[{name}] max_diff={diff.max():.6f} mean_diff={diff.mean():.6f} PCC={pcc_val:.6f}")


def main():
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_flow = ref_model.flow
    ref_flow.eval()

    batch_size = 1
    token_len2 = 5
    token_len1 = 5
    mel_len1 = 10

    token = torch.randint(0, 4096, (batch_size, token_len2), dtype=torch.int32)
    token_len_t = torch.tensor([token_len2], dtype=torch.int32)
    prompt_token = torch.randint(0, 4096, (batch_size, token_len1), dtype=torch.int32)
    prompt_feat = torch.randn(batch_size, mel_len1, ref_flow.output_size, dtype=torch.float32)
    embedding = torch.randn(batch_size, 192, dtype=torch.float32)
    flow_cache = torch.zeros(batch_size, 80, 0, 2, dtype=torch.float32)

    # Reference inference
    with torch.no_grad():
        ref_feat, _ = ref_flow.inference(
            token=token,
            token_len=token_len_t,
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor([token_len1], dtype=torch.int32),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([mel_len1], dtype=torch.int32),
            embedding=embedding,
            flow_cache=flow_cache,
        )

    # TT inference
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    tt_flow = TtCosyVoiceFlow(device, state_dict=ref_flow.state_dict(), ref_flow=ref_flow)

    # Manually step through to isolate differences
    token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
    token_all = torch.cat([prompt_token, token], dim=1)
    token_len_all = torch.tensor([token_len1 + token_len2], dtype=torch.int32)
    mask = (~make_pad_mask(token_len_all)).unsqueeze(-1).to(embedding.dtype)
    token_emb = tt_flow.input_embedding(torch.clamp(token_all, min=0)) * mask

    # Compare token_emb
    with torch.no_grad():
        ref_mask = (~make_pad_mask(token_len_all)).unsqueeze(-1).to(ref_flow.input_embedding.weight.dtype)
        ref_token_emb = ref_flow.input_embedding(torch.clamp(token_all, min=0)) * ref_mask
    comp_tensors(ref_token_emb, token_emb, "token_emb")

    # Compare encoder output
    token_tt = ttnn.from_torch(token_emb, layout=ttnn.TILE_LAYOUT, device=device)
    h, _ = tt_flow.encoder(token_tt, token_len=token_len_all)
    h_cpu = ttnn.to_torch(h).float()

    with torch.no_grad():
        ref_h, _ = ref_flow.encoder(token_emb, token_len_all)
    comp_tensors(ref_h, h_cpu, "encoder_out")

    # Compare after encoder_proj
    h_proj = tt_flow.encoder_proj(h_cpu)
    with torch.no_grad():
        ref_h_proj = ref_flow.encoder_proj(ref_h)
    comp_tensors(ref_h_proj, h_proj, "encoder_proj")

    # Compare after regulator
    mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / 50 * 22050 / 256)
    h_reg, _ = tt_flow.length_regulator.inference(
        h_proj[:, :token_len1], h_proj[:, token_len1:], mel_len1, mel_len2, 50
    )
    h_reg_torch = ttnn.to_torch(h_reg).float() if isinstance(h_reg, ttnn.Tensor) else h_reg.float()

    with torch.no_grad():
        ref_h_reg, _ = ref_flow.length_regulator.inference(
            ref_h[:, :token_len1], ref_h[:, token_len1:], mel_len1, mel_len2, 50
        )
    comp_tensors(ref_h_reg, h_reg_torch, "regulator_out")

    # Compare final decoder input (mu)
    conds = torch.zeros([1, mel_len1 + mel_len2, 80], device=embedding.device, dtype=torch.float32)
    conds[:, :mel_len1] = prompt_feat
    conds = conds.transpose(1, 2)
    mask_cpu = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).float()
    embedding_norm = F.normalize(embedding, dim=1)
    embedding_proj = tt_flow.spk_embed_affine_layer(embedding_norm)

    with torch.no_grad():
        ref_embedding_norm = F.normalize(embedding, dim=1)
        ref_embedding_proj = ref_flow.spk_embed_affine_layer(ref_embedding_norm)
    comp_tensors(ref_embedding_proj, embedding_proj, "spk_embed")

    # Run TT decoder
    tt_feat, _ = tt_flow.decoder(
        mu=ttnn.from_torch(h_reg_torch.transpose(1, 2).contiguous(), layout=ttnn.TILE_LAYOUT, device=device),
        mask=ttnn.from_torch(mask_cpu.unsqueeze(1), layout=ttnn.TILE_LAYOUT, device=device),
        n_timesteps=10,
        spks=ttnn.from_torch(embedding_proj, layout=ttnn.TILE_LAYOUT, device=device),
        cond=ttnn.from_torch(conds, layout=ttnn.TILE_LAYOUT, device=device),
        prompt_len=mel_len1,
        cache=flow_cache,
    )

    # Run reference decoder
    with torch.no_grad():
        ref_feat, _ = ref_flow.decoder(
            mu=ref_h_reg.transpose(1, 2).contiguous(),
            mask=mask_cpu.unsqueeze(1),
            n_timesteps=10,
            spks=ref_embedding_proj,
            cond=conds,
            prompt_len=mel_len1,
            cache=flow_cache,
        )

    tt_feat_torch = ttnn.to_torch(tt_feat).float()
    print(f"\n--- Final comparison ---")
    print(f"Ref feat shape: {ref_feat.shape}, TT feat shape: {tt_feat_torch.shape}")
    comp_tensors(ref_feat, tt_feat_torch, "final_decoder_out")

    # Full end-to-end
    print(f"\n--- Full end-to-end ---")
    comp_tensors(ref_feat, ref_feat)  # sanity
    comp_tensors(ref_feat, tt_feat_torch[:, :, mel_len1:])

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
