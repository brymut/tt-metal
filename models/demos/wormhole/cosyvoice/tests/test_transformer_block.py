# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test the full TtBasicTransformerBlock output vs reference at different t values.
This isolates whether the per-t PCC drop is in the transformer blocks (self-attn
or FF) or the up path / final proj.
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
from models.demos.wormhole.cosyvoice.tt.cosyvoice_unet import TtBasicTransformerBlock


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


def test_transformer_block_vs_reference(device, reference_model):
    """Compare TtBasicTransformerBlock output vs reference for the first
    down-block's first transformer block."""
    from matcha.models.components.decoder import SinusoidalPosEmb

    sp = SinusoidalPosEmb(dim=320)

    estimator = reference_model.flow.decoder.estimator
    # First down-block: down_blocks.0.1.0 (index 1 = tbs, index 0 = first tb)
    ref_tb = estimator.down_blocks[0][1][0]
    ref_time_mlp = estimator.time_mlp

    estimator_sd = estimator.state_dict()
    tt_tb = TtBasicTransformerBlock(
        device,
        estimator_sd,
        "down_blocks.0.1.0",
        dim=256,
        num_heads=8,
        attention_head_dim=64,
        activation_fn="gelu",
    )

    torch.manual_seed(0)
    B = 1
    T = 18
    C = 256

    x = torch.randn(B, T, C, dtype=torch.float32)
    # No attn bias (mask is all-ones in the UNet test)
    attn_bias = None

    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t_values = t_span[:-1]

    print("\n" + "=" * 72)
    print(f"{'t':>10} | {'PCC':>10} | {'max|Δ|':>10}")
    print("-" * 72)

    for t_val in t_values.tolist():
        t_t = torch.tensor([t_val], dtype=torch.float32)

        # Reference: run the transformer block with the same input
        with torch.no_grad():
            ref_out = ref_tb(x)

        # TT: run the transformer block with the same input
        x_tt = ttnn.from_torch(x.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        with torch.no_grad():
            tt_out_tt = tt_tb.forward(x_tt, attn_bias_tt=attn_bias)
        tt_out = ttnn.to_torch(tt_out_tt).float()

        pcc_pass, pcc_value = comp_pcc(ref_out, tt_out, pcc=0.90)
        max_abs = (ref_out - tt_out).abs().max().item()
        print(f"{t_val:>10.4f} | {pcc_value:>10.4f} | {max_abs:>10.4f}")

    print("=" * 72)
