# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Inspect UNet output magnitudes at t=0 vs t=0.5 and the per-block
mlp(time_emb) contribution to localize where the per-t error amplification
happens.
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

from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


def test_inspect_unet_magnitudes(device, reference_model):
    """Print per-t UNet output magnitudes and the per-block mlp(time_emb)
    contribution. Goal: see if t=0 is a 'time-embedding-dominated' regime
    where small time-emb errors get amplified.
    """
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

    # Reference: compute the per-resnet mlp(time_emb) at t=0 and t=0.5
    from matcha.models.components.decoder import SinusoidalPosEmb

    sp = SinusoidalPosEmb(dim=320)

    print("\n=== Time embedding path ===")
    for t_val in [0.0, 0.5, 1.0]:
        t_t = torch.tensor([t_val], dtype=torch.float32)
        ref_emb = sp(t_t, scale=1000.0).float()
        ref_time_out = estimator.time_mlp(ref_emb)  # [1, 1024]
        # First down-block resnet mlp
        ref_resnet = estimator.down_blocks[0][0]  # ResnetBlock1D
        ref_mlp_out = ref_resnet.mlp(ref_time_out)  # Mish + Linear
        print(
            f"t={t_val:.2f}: time_emb std={ref_time_out.std():.3f}  "
            f"resnet0.mlp_out std={ref_mlp_out.std():.3f}  "
            f"|resnet0.mlp_out mean|={ref_mlp_out.abs().mean():.3f}"
        )

    # Now run the reference UNet and print the output magnitude
    print("\n=== Reference UNet output magnitude ===")
    for t_val in [0.0, 0.5, 1.0]:
        t_t = torch.tensor([t_val], dtype=torch.float32)
        with torch.no_grad():
            ref_out = estimator(x=x, mask=mask, mu=mu, t=t_t, spks=spks, cond=cond, streaming=False)
        print(
            f"t={t_val:.2f}: UNet out std={ref_out.std():.3f}  "
            f"mean abs={ref_out.abs().mean():.3f}  "
            f"max abs={ref_out.abs().max():.3f}"
        )
