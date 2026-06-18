# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test the full TtResnetBlock1D output vs reference at different t values.
This isolates whether the per-t PCC drop is in the ResnetBlock or downstream
(transformer blocks, up path, final proj).
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
from models.demos.wormhole.cosyvoice.tt.cosyvoice_unet import TtResnetBlock1D, TtTimeEmbeddings


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


def test_resnet_block_vs_reference(device, reference_model):
    """Compare TtResnetBlock1D output vs reference for the first down-block resnet."""
    estimator = reference_model.flow.decoder.estimator
    ref_resnet = estimator.down_blocks[0][0]  # First down-block ResnetBlock1D
    ref_time_mlp = estimator.time_mlp

    # Build TtResnetBlock1D
    # The estimator's own state_dict() uses keys like "down_blocks.0.0.mlp.1.weight"
    # (no "decoder.estimator." prefix). TtResnetBlock1D expects base.mlp.1.weight.
    estimator_sd = estimator.state_dict()
    tt_resnet = TtResnetBlock1D(
        device,
        estimator_sd,
        "down_blocks.0.0",
        dim=320,
        dim_out=256,
        time_emb_dim=1024,
        groups=8,
    )

    # Build TtTimeEmbeddings
    tt_te = TtTimeEmbeddings(
        device,
        estimator_sd,
        "time_mlp",
        in_channels=320,
        time_embed_dim=1024,
    )

    torch.manual_seed(0)
    B = 1
    T = 18
    in_ch = 320
    out_ch = 256

    x = torch.randn(B, in_ch, T, dtype=torch.float32)
    mask = torch.ones(B, 1, T, dtype=torch.float32)

    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t_values = t_span[:-1]

    print("\n" + "=" * 72)
    print(f"{'t':>10} | {'PCC':>10} | {'max|Δ|':>10}")
    print("-" * 72)

    for t_val in t_values.tolist():
        t_t = torch.tensor([t_val], dtype=torch.float32)

        # Reference
        with torch.no_grad():
            ref_emb = ref_time_mlp(
                torch.cat(
                    (
                        torch.sin(
                            t_t
                            * 1000
                            * torch.exp(
                                -torch.arange(160, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / 159)
                            ).unsqueeze(0)
                        ),
                        torch.cos(
                            t_t
                            * 1000
                            * torch.exp(
                                -torch.arange(160, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / 159)
                            ).unsqueeze(0)
                        ),
                    ),
                    dim=-1,
                )
            )
            ref_out = ref_resnet(x, mask, ref_emb)

        # TT
        x_tt = ttnn.from_torch(
            x.unsqueeze(1).transpose(2, 3).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        mask_tt = ttnn.from_torch(
            mask.unsqueeze(-1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        time_emb_tt = tt_te.forward(t_t)

        with torch.no_grad():
            tt_out_tt = tt_resnet.forward(x_tt, mask_tt, time_emb_tt, B, T)
        tt_out = ttnn.to_torch(tt_out_tt).float().squeeze(1).transpose(1, 2)  # [B, out_ch, T]

        pcc_pass, pcc_value = comp_pcc(ref_out, tt_out, pcc=0.90)
        max_abs = (ref_out - tt_out).abs().max().item()
        print(f"{t_val:>10.4f} | {pcc_value:>10.4f} | {max_abs:>10.4f}")

    print("=" * 72)
