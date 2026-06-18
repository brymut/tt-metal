# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate the TtResnetBlock1D's mlp(time_emb) path vs reference.
Mish + Linear applied to the time embedding at each t value.
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


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


def test_resnet_mlp_path(device, reference_model):
    """Compare TtResnetBlock1D's mlp(time_emb) = Mish(time_emb) @ W + b
    vs the reference, for the first down-block resnet, at the cosine t values."""
    from matcha.models.components.decoder import SinusoidalPosEmb

    sp = SinusoidalPosEmb(dim=320)

    # Reference: time_mlp + first down-block resnet.mlp
    ref_time_mlp = reference_model.flow.decoder.estimator.time_mlp
    ref_resnet = reference_model.flow.decoder.estimator.down_blocks[0][0]
    ref_mlp = ref_resnet.mlp  # nn.Sequential(nn.Mish(), nn.Linear(1024, 256))

    # Build the TT version of the mlp path

    # TtResnetBlock1D expects a state_dict with mlp.1.weight, mlp.1.bias
    # and block1.block.0.weight, etc. We only need the mlp path here.
    # Build a minimal TtResnetBlock1D just to get the mlp linear weights.
    sd = {f"mlp.1.{k}": v for k, v in ref_mlp[1].state_dict().items()}

    # We need to use the TtResnetBlock1D's mlp linear directly. The simplest
    # way is to instantiate TtResnetBlock1D and call it with a dummy input.
    # But TtResnetBlock1D needs the full state_dict. Let's just replicate the
    # mlp linear manually.
    from models.demos.wormhole.cosyvoice.tt.cosyvoice_unet import TtTimeEmbeddings

    tt_te = TtTimeEmbeddings(
        device,
        {f"time_mlp.{k}": v for k, v in ref_time_mlp.state_dict().items()},
        "time_mlp",
        in_channels=320,
        time_embed_dim=1024,
    )
    mlp_weight = ttnn.from_torch(
        ref_mlp[1].weight.T.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    mlp_bias = ttnn.from_torch(
        ref_mlp[1].bias.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t_values = t_span[:-1]

    print("\n" + "=" * 72)
    print(f"{'t':>10} | {'PCC':>10} | {'max|Δ|':>10}")
    print("-" * 72)

    for t_val in t_values.tolist():
        t_t = torch.tensor([t_val], dtype=torch.float32)
        with torch.no_grad():
            ref_emb = sp(t_t, scale=1000.0).float()
            ref_time_out = ref_time_mlp(ref_emb)  # [1, 1024]
            ref_mlp_out = ref_mlp(ref_time_out)  # [1, 256]

        # TT: time_embeddings -> mish -> linear
        tt_time_out = tt_te.forward(t_t)  # [1, 1024]
        tt_time_act = ttnn.mish(tt_time_out)
        tt_mlp_out = ttnn.linear(
            tt_time_act,
            mlp_weight,
            bias=mlp_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        tt_mlp_t = ttnn.to_torch(tt_mlp_out).float()

        pcc_pass, pcc_value = comp_pcc(ref_mlp_out, tt_mlp_t, pcc=0.99)
        max_abs = (ref_mlp_out - tt_mlp_t).abs().max().item()
        print(f"{t_val:>10.4f} | {pcc_value:>10.6f} | {max_abs:>10.4f}")

    print("=" * 72)
