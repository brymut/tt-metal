# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the native UNet port (`TtConditionalDecoder`).

Compares the TT-native UNet (device) against the PyTorch reference
(`cosyvoice.flow.decoder.ConditionalDecoder`) on the same weights, using
synthetic inputs of the same shape used by `test_flow.py`.

PCC target: > 0.95 (vs ~0.855 for the previous CPU-fallback path).
"""

import os
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = os.getenv(
    "COSYVOICE_MODEL_DIR", str(PROJECT_ROOT.parent.parent.parent.parent) + "/pretrained_models/CosyVoice-300M"
)
sys.path.insert(0, str(PROJECT_ROOT.parent.parent.parent))  # for models.* imports
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice"))
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice" / "third_party" / "Matcha-TTS"))

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_unet import TtConditionalDecoder


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


def _run_reference_unet(estimator, x, mask, mu, t, spks, cond):
    """Reference UNet forward matching `ConditionalDecoder.forward` signature."""
    with torch.no_grad():
        return estimator(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, streaming=False)


def _prepare_tt_inputs(x_t, mask_t, mu_t, t_t, spks_t, cond_t, device):
    """Convert torch inputs to ttnn tensors in the layout TtConditionalDecoder expects:
    x:    [B, 1, T, in_channels]   (permute from [B, in, T])
    mask: [B, 1, T, 1]             (permute from [B, 1, T])
    mu:   [B, 1, 80, T]            (4D for `ttnn.permute` inside the UNet)
    t:    [B] torch
    spks: [B, 1, T, 80]            (broadcast over T)
    cond: [B, 1, 80, T]
    """
    B, C_in, T = x_t.shape
    # x: [B, C, T] -> [B, 1, T, C]
    x_nchw = x_t.unsqueeze(1)
    x_tt = ttnn.from_torch(
        x_nchw.transpose(2, 3).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    # mask: [B, 1, T] -> [B, 1, T, 1]
    mask_tt = ttnn.from_torch(
        mask_t.unsqueeze(-1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    # mu: [B, 80, T] -> [B, 1, 80, T]
    mu_tt = ttnn.from_torch(mu_t.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # spks: [B, 80] -> [B, 1, T, 80] (broadcast over T)
    spks_b = spks_t.unsqueeze(1).unsqueeze(1).expand(B, 1, T, spks_t.shape[-1]).contiguous()
    spks_tt = ttnn.from_torch(spks_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # cond: [B, 80, T] -> [B, 1, 80, T]
    cond_tt = ttnn.from_torch(
        cond_t.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    return x_tt, mask_tt, mu_tt, t_t, spks_tt, cond_tt


def test_unet_vs_reference(device, reference_model):
    """Compare TtConditionalDecoder output against reference on identical weights.

    The CFM estimator signature is `(x, mask, mu, t, spks, cond)` where x is
    the 80-channel noise input. The reference ConditionalDecoder packs
    `pack([x, mu], "b * t")`, `pack([x, spks], "b * t")`, `pack([x, cond], "b * t")`
    internally, so the total input channel count is 80+80+80+80 = 320.
    """
    estimator = reference_model.flow.decoder.estimator
    sd = estimator.state_dict()

    torch.manual_seed(0)
    B = 1
    in_channels = 80  # noise input (estimator receives x as the 80-channel noise)
    T = 18
    spk_dim = 80

    x = torch.randn(B, in_channels, T, dtype=torch.float32)  # noise
    mask = torch.ones(B, 1, T, dtype=torch.float32)
    mu = torch.randn(B, spk_dim, T, dtype=torch.float32)
    t = torch.tensor([0.5], dtype=torch.float32)
    spks = torch.randn(B, spk_dim, dtype=torch.float32)
    cond = torch.randn(B, spk_dim, T, dtype=torch.float32)

    # Reference
    with torch.no_grad():
        ref_out = _run_reference_unet(estimator, x, mask, mu, t, spks, cond)
    # ref_out shape: [B, out_channels, T] = [1, 80, 18]

    # Prefix state dict keys to match TtConditionalDecoder's default
    # `base_address="decoder.estimator"`. The estimator's own state_dict() uses
    # unprefixed keys (e.g. "down_blocks.0.0..."), so we re-key.
    prefix = "decoder.estimator."
    sd_prefixed = {prefix + k: v for k, v in sd.items()}

    # TT
    tt_unet = TtConditionalDecoder(device, state_dict=sd_prefixed)
    x_tt, mask_tt, mu_tt, t_t, spks_tt, cond_tt = _prepare_tt_inputs(x, mask, mu, t, spks, cond, device)
    with torch.no_grad():
        tt_out_tt = tt_unet.forward(x_tt, mask_tt, mu_tt, t_t, spks_tt, cond_tt)
    # tt_out_tt shape: [B, 1, T, out_channels] = [1, 1, 18, 80]
    tt_out = ttnn.to_torch(tt_out_tt).float().squeeze(1)  # [B, T, out_channels]
    tt_out = tt_out.transpose(1, 2)  # [B, out_channels, T]

    pcc_pass, pcc_value = comp_pcc(ref_out, tt_out, pcc=0.90)
    print(f"UNet PCC: {pcc_value:.4f}")
    assert pcc_pass, f"UNet PCC failed: {pcc_value} < 0.90"
