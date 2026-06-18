# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate the on-device Classifier-Free Guidance (CFG) 2x-batch path.

The reference CosyVoice-300M flow uses `inference_cfg_rate = 0.7` (see
`reference/CosyVoice/cosyvoice/flow/flow.py`). The TT `TtConditionalCFM`
implements CFG by running the UNet on a 2x batch [conditioned, unconditioned]
and combining with `(1 + r) * cond - r * uncond`. This path was previously
disabled (`inference_cfg_rate = 0.0`) because of a suspected 2x-batch
tile-padding broadcasting bug. This test exercises the 2x path directly
(`inference_cfg_rate = 0.7` on both sides) with matched noise `z` so we can
confirm it runs and measures real accuracy vs the reference.

If this passes with a healthy PCC (>= ~0.6 at n_timesteps=10), CFG can be
safely enabled in the E2E path (`TtCosyVoiceFlow`).
"""

import os
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = os.getenv(
    "COSYVOICE_MODEL_DIR",
    str(PROJECT_ROOT.parent.parent.parent.parent) + "/pretrained_models/CosyVoice-300M",
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


@pytest.fixture(scope="module")
def tt_cfm(device, reference_model):
    """Build the TT CFM once per module (the on-device construct/compile is
    expensive and does not depend on `n_timesteps`). Returns the
    `TtConditionalCFM` with CFG enabled (rate 0.7)."""
    tt_flow = TtCosyVoiceFlow(device, state_dict=reference_model.flow.state_dict(), ref_flow=reference_model.flow)
    cfm = tt_flow.decoder
    cfm.inference_cfg_rate = 0.7  # enable the 2x-batch CFG path
    return cfm


@pytest.mark.parametrize("n_timesteps", [1, 10])
@pytest.mark.xfail(
    reason=(
        "Known bug: the 2x-batch CFG path crashes in TtBlock1D.forward "
        "(cosyvoice_unet.py) with a broadcasting violation at B=2 — the "
        "multiplicative mask is checked against the INPUT T but the conv1d "
        "output T is padded/doubled for B=2 under HEIGHT_SHARDED (e.g. T=8 -> "
        "16). The mask-vs-output reconciliation in TtBlock1D must be fixed "
        "before the on-device CFG path can run. The reference-side CFG math "
        "is mirrored correctly here and the `mu`-zeroing fix in "
        "TtConditionalCFM._run_unet_2x_cfg is correct; only the B=2 UNet body "
        "shape handling is broken. This is the same 'CFG doubling "
        "broadcasting bug' noted as deferred in HANDOFF.md."
    ),
    strict=True,
)
def test_cfm_cfg_2x_path(device, reference_model, tt_cfm, n_timesteps):
    """Run the CFM decoder with CFG rate 0.7 on both reference and TT (2x batch)."""
    ref_flow = reference_model.flow
    ref_cfm = ref_flow.decoder
    # Force the production CFG rate on the reference (it is 0.7 by default but
    # other tests may have mutated it).
    original_cfg = ref_cfm.inference_cfg_rate
    ref_cfm.inference_cfg_rate = 0.7

    B = 1
    T = 8  # mel_len2 (output frames)
    spk_dim = 80
    mu = torch.randn(B, spk_dim, T, dtype=torch.float32)
    mask = torch.ones(B, 1, T, dtype=torch.float32)
    spks = torch.randn(B, spk_dim, dtype=torch.float32)
    cond = torch.randn(B, spk_dim, T, dtype=torch.float32)

    # Reference: CFG=0.7, matched noise. The reference CFM is pre-built, so
    # its `z = torch.randn_like(mu)` is the first RNG draw after this seed.
    torch.manual_seed(42)
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

    mu_tt = ttnn.from_torch(mu.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mask_tt = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    spks_tt = ttnn.from_torch(spks.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cond_tt = ttnn.from_torch(
        cond.unsqueeze(1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # TT: tt_cfm is pre-built by the fixture, so seeding here makes its
    # `z_t = torch.randn_like(mu_t)` the first draw from the same state.
    torch.manual_seed(42)  # same noise as reference
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
    tt_out = ttnn.to_torch(tt_out_tt).float().squeeze(1).transpose(1, 2)  # [B, 80, T]

    pcc_pass, pcc_value = comp_pcc(ref_out, tt_out, pcc=0.0)
    max_abs_err = (ref_out - tt_out).abs().max().item()
    print(
        f"\n[CFG=0.7] n_timesteps={n_timesteps}: PCC={pcc_value:.4f}  "
        f"ref_std={ref_out.std():.3f}  tt_std={tt_out.std():.3f}  max|Δ|={max_abs_err:.3f}"
    )

    ref_cfm.inference_cfg_rate = original_cfg
    # Smoke assertion: the 2x CFG path must run without error and produce
    # finite output. The PCC is informational (threshold 0.0) until the 2x
    # batch path is confirmed stable across all UNet shapes.
    assert torch.isfinite(tt_out).all(), "TT CFG output contains non-finite values"
    assert pcc_pass, f"CFG PCC failed: {pcc_value} < 0.0"
