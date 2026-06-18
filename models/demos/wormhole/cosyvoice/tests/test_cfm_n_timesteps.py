# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate whether the E2E flow PCC drop is purely from per-call UNet error
accumulation or a separate solver-integration bug.

Strategy: call the CFM decoder (reference + TT) directly with n_timesteps in
{1, 2, 10} on the same inputs. If PCC at n_timesteps=1 matches the per-call
UNet PCC (~0.74 at t=0), the E2E drop is purely per-call error accumulation.
If n_timesteps=1 is much better, there's a separate solver bug.

We feed both the reference and the TT decoder the same mu/mask/spks/cond and
the same initial noise z (via torch.manual_seed).
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
from models.demos.wormhole.cosyvoice.tt.cosyvoice_flow import TtCosyVoiceFlow


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


@pytest.mark.parametrize("n_timesteps", [1, 2, 10])
def test_cfm_decoder_n_timesteps(device, reference_model, n_timesteps):
    """Run the CFM decoder (reference + TT) with varying n_timesteps."""
    ref_flow = reference_model.flow
    ref_cfm = ref_flow.decoder
    # Disable CFG on both paths
    original_cfg = ref_cfm.inference_cfg_rate
    ref_cfm.inference_cfg_rate = 0.0

    # Synthetic inputs matching the shape used by test_flow.py
    B = 1
    T = 8  # mel_len2 (output frames)
    spk_dim = 80
    mu = torch.randn(B, spk_dim, T, dtype=torch.float32)
    mask = torch.ones(B, 1, T, dtype=torch.float32)
    spks = torch.randn(B, spk_dim, dtype=torch.float32)
    cond = torch.randn(B, spk_dim, T, dtype=torch.float32)

    # Reference: call the reference CFM decoder directly
    torch.manual_seed(42)  # so the noise z is the same in both paths
    with torch.no_grad():
        ref_out, _ = ref_cfm(
            mu=mu,
            mask=mask,
            n_timesteps=n_timesteps,
            spks=spks,
            cond=cond,
            prompt_len=0,
            cache=torch.zeros(1, 80, 0, 2, dtype=torch.float32),
        )
    # ref_out shape: [B, 80, T]

    # TT: build the TT CFM and call it
    tt_flow = TtCosyVoiceFlow(device, state_dict=ref_flow.state_dict(), ref_flow=ref_flow)
    tt_cfm = tt_flow.decoder  # already has CFG=0

    # Convert inputs to ttnn in the layouts the TT CFM expects
    torch.manual_seed(42)  # same noise
    mu_tt = ttnn.from_torch(
        mu.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )  # [B, 1, 80, T]
    mask_tt = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # [B, 1, T]
    spks_tt = ttnn.from_torch(
        spks.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )  # [B, 1, 80]
    cond_tt = ttnn.from_torch(
        cond.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )  # [B, 1, 80, T]

    with torch.no_grad():
        tt_out_tt, _ = tt_cfm(
            mu=mu_tt,
            mask=mask_tt,
            n_timesteps=n_timesteps,
            spks=spks_tt,
            cond=cond_tt,
            prompt_len=0,
            cache=torch.zeros(1, 80, 0, 2, dtype=torch.float32),
        )
    # tt_out_tt shape: [B, 1, T, 80] (ttnn layout)
    tt_out = ttnn.to_torch(tt_out_tt).float().squeeze(1)  # [B, T, 80]
    tt_out = tt_out.transpose(1, 2)  # [B, 80, T]

    pcc_pass, pcc_value = comp_pcc(ref_out, tt_out, pcc=0.0)
    ref_std = ref_out.std().item()
    tt_std = tt_out.std().item()
    max_abs_err = (ref_out - tt_out).abs().max().item()
    print(
        f"\nn_timesteps={n_timesteps}: PCC={pcc_value:.4f}  "
        f"ref_std={ref_std:.3f}  tt_std={tt_std:.3f}  max|Δ|={max_abs_err:.3f}"
    )

    ref_cfm.inference_cfg_rate = original_cfg
    # No assertion — this is a diagnostic.
