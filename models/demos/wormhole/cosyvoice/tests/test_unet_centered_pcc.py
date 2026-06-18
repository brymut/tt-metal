# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Check whether the per-t UNet PCC drop is a DC-offset issue by comparing
centered (mean-subtracted) PCC. If centered PCC recovers to ~0.99, the
problem is a small bias/offset discrepancy that gets amplified at t=0.
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
sys.path.insert(0, str(PROJECT_ROOT.parent.parent.parent))
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice"))
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice" / "third_party" / "Matcha-TTS"))

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_unet import TtConditionalDecoder


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


def _centered_pcc(a, b):
    a_c = a - a.mean()
    b_c = b - b.mean()
    return torch.tensor(
        torch.nn.functional.cosine_similarity(a_c.flatten().unsqueeze(0), b_c.flatten().unsqueeze(0)).item()
    )


def test_unet_centered_pcc(device, reference_model):
    estimator = reference_model.flow.decoder.estimator
    sd = estimator.state_dict()

    torch.manual_seed(0)
    B = 1
    in_channels = 80
    T = 18
    spk_dim = 80

    x = torch.randn(B, in_channels, T, dtype=torch.float32)
    mask = torch.ones(B, 1, T, dtype=torch.float32)
    mu = torch.randn(B, spk_dim, T, dtype=torch.float32)
    spks = torch.randn(B, spk_dim, dtype=torch.float32)
    cond = torch.randn(B, spk_dim, T, dtype=torch.float32)

    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t_values = t_span[:-1]

    prefix = "decoder.estimator."
    sd_prefixed = {prefix + k: v for k, v in sd.items()}
    tt_unet = TtConditionalDecoder(device, state_dict=sd_prefixed)

    print("\n" + "=" * 84)
    print(f"{'t':>8} | {'raw PCC':>10} | {'centered PCC':>14} | {'ref mean':>10} | {'TT mean':>10} | {'mean Δ':>10}")
    print("-" * 84)

    for t_val in t_values.tolist():
        t_t = torch.tensor([t_val], dtype=torch.float32)
        with torch.no_grad():
            ref_out = estimator(x=x, mask=mask, mu=mu, t=t_t, spks=spks, cond=cond, streaming=False)

        x_nchw = x.unsqueeze(1)
        x_tt = ttnn.from_torch(
            x_nchw.transpose(2, 3).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        mask_tt = ttnn.from_torch(
            mask.unsqueeze(-1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        mu_tt = ttnn.from_torch(
            mu.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        spks_b = spks.unsqueeze(1).unsqueeze(1).expand(B, 1, T, spks.shape[-1]).contiguous()
        spks_tt = ttnn.from_torch(spks_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cond_tt = ttnn.from_torch(
            cond.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        with torch.no_grad():
            tt_out_tt = tt_unet.forward(x_tt, mask_tt, mu_tt, t_t, spks_tt, cond_tt)
        tt_out = ttnn.to_torch(tt_out_tt).float().squeeze(1).transpose(1, 2)

        pcc_pass, pcc_value = comp_pcc(ref_out, tt_out, pcc=0.90)
        centered = _centered_pcc(ref_out, tt_out)
        ref_mean = ref_out.mean().item()
        tt_mean = tt_out.mean().item()
        mean_diff = (ref_out - tt_out).mean().item()
        print(
            f"{t_val:>8.4f} | {pcc_value:>10.4f} | {centered.item():>14.4f} | {ref_mean:>10.3f} | {tt_mean:>10.3f} | {mean_diff:>10.3f}"
        )

    print("=" * 84)
