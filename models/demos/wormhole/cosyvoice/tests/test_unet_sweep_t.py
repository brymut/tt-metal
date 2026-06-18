# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Localize the E2E flow PCC drop by sweeping the UNet at all t values used
by the cosine-scheduled Euler solver (t = 0, 0.024, 0.095, ..., 0.976, 1.0).

Hypothesis 1 (from HANDOFF.md): per-step UNet output at extreme t values (0, 1)
is much worse than at t=0.5.
Hypothesis 2: bug in TtTimeEmbeddings (bf16 silu of extreme sinusoid values).
Hypothesis 3: numerical drift in the Euler solver (ttnn.add rounding).
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


def _run_reference_unet(estimator, x, mask, mu, t, spks, cond):
    with torch.no_grad():
        return estimator(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, streaming=False)


def _prepare_tt_inputs(x_t, mask_t, mu_t, t_t, spks_t, cond_t, device):
    B, C_in, T = x_t.shape
    x_nchw = x_t.unsqueeze(1)
    x_tt = ttnn.from_torch(
        x_nchw.transpose(2, 3).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    mask_tt = ttnn.from_torch(
        mask_t.unsqueeze(-1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    mu_tt = ttnn.from_torch(mu_t.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    spks_b = spks_t.unsqueeze(1).unsqueeze(1).expand(B, 1, T, spks_t.shape[-1]).contiguous()
    spks_tt = ttnn.from_torch(spks_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cond_tt = ttnn.from_torch(
        cond_t.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    return x_tt, mask_tt, mu_tt, t_t, spks_tt, cond_tt


def test_unet_sweep_t_values(device, reference_model):
    """Run TtConditionalDecoder at every t used by the cosine-scheduled Euler solver."""
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

    # Build the cosine schedule used by the solver
    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t_values = t_span[:-1]  # the t fed to each UNet call (step 1..N uses t_span[i-1])

    # Prefix state dict keys for TtConditionalDecoder's default base_address
    prefix = "decoder.estimator."
    sd_prefixed = {prefix + k: v for k, v in sd.items()}
    tt_unet = TtConditionalDecoder(device, state_dict=sd_prefixed)

    print("\n" + "=" * 72)
    print(f"{'t':>10} | {'PCC':>10} | {'cos(sched)':>11} | {'sin_max@t':>10}")
    print("-" * 72)

    results = []
    for t_val in t_values.tolist():
        t_t = torch.tensor([t_val], dtype=torch.float32)
        with torch.no_grad():
            ref_out = _run_reference_unet(estimator, x, mask, mu, t_t, spks, cond)

        x_tt, mask_tt, mu_tt, _, spks_tt, cond_tt = _prepare_tt_inputs(x, mask, mu, t_t, spks, cond, device)
        with torch.no_grad():
            tt_out_tt = tt_unet.forward(x_tt, mask_tt, mu_tt, t_t, spks_tt, cond_tt)
        tt_out = ttnn.to_torch(tt_out_tt).float().squeeze(1)
        tt_out = tt_out.transpose(1, 2)

        pcc_pass, pcc_value = comp_pcc(ref_out, tt_out, pcc=0.90)
        # Diagnostic: what does the sinusoidal embedding look like at this t?
        import math

        scale = 1000.0
        half_dim = in_channels // 2
        emb_freq = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_freq)
        sin_arg = scale * t_val * freqs
        sin_max = sin_arg.max().item()
        results.append((t_val, pcc_value, pcc_pass))
        print(f"{t_val:>10.4f} | {pcc_value:>10.4f} | {t_val:>11.4f} | {sin_max:>10.2f}")

    print("=" * 72)
    # Highlight: did the extreme-t calls (t→0 or t→1) collapse?
    t0_pcc = results[0][1]
    t1_pcc = results[-1][1]
    t_mid_pcc = results[len(results) // 2][1]
    print(f"\nSummary: PCC at t~0   = {t0_pcc:.4f}")
    print(f"         PCC at t~0.5 = {t_mid_pcc:.4f}")
    print(f"         PCC at t~1   = {t1_pcc:.4f}")

    # The standalone test uses t=0.5 and expects PCC > 0.90. We use a permissive
    # threshold here so the diagnostic always runs; assertions are about logging.
    assert t_mid_pcc > 0.85, f"mid-t PCC collapsed: {t_mid_pcc}"
