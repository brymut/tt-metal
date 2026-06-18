# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate TtTimeEmbeddings vs the reference `TimestepEmbedding` at the
cosine-scheduled t values.

Hypothesis: the time embedding itself is the source of the per-t PCC drift
seen in `test_unet_sweep_t.py`. At t=0 the sinusoidal input is [0,...,0,1,...,1]
(all sin=0, all cos=1), so the linear_1 bias dominates and the silu output
is at a constant nonzero value; the bf16 path may quantise the constant
silu output differently from the reference fp32 path.
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
from models.demos.wormhole.cosyvoice.tt.cosyvoice_unet import TtTimeEmbeddings


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


def test_time_embeddings_sweep_t(device, reference_model):
    """Compare TtTimeEmbeddings vs the reference time_mlp at the same t values
    used by the cosine-scheduled Euler solver."""
    ref_time_mlp = reference_model.flow.decoder.estimator.time_mlp
    # ref_time_mlp is `TimestepEmbedding(in_channels=320, time_embed_dim=1024, act_fn='silu')`

    # TtTimeEmbeddings reads from state_dict via base_address="time_mlp"
    # and looks for time_mlp.linear_1.weight etc.
    sd = {f"time_mlp.{k}": v for k, v in ref_time_mlp.state_dict().items()}

    tt_te = TtTimeEmbeddings(
        device,
        sd,
        "time_mlp",
        in_channels=320,
        time_embed_dim=1024,
    )

    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t_values = t_span[:-1]

    print("\n" + "=" * 72)
    print(f"{'t':>10} | {'PCC':>10} | {'sin_max@t':>10} | {'max|Δ|':>12}")
    print("-" * 72)

    for t_val in t_values.tolist():
        t_t = torch.tensor([t_val], dtype=torch.float32)
        with torch.no_grad():
            # Reference: sinusoidal_pos_emb (on CPU, dim=320) then time_mlp
            from matcha.models.components.decoder import SinusoidalPosEmb

            sp = SinusoidalPosEmb(dim=320)
            ref_emb = sp(t_t, scale=1000.0).float()  # [1, 320]
            ref_out = ref_time_mlp(ref_emb)  # [1, 1024]

        tt_out_tt = tt_te.forward(t_t)
        tt_out = ttnn.to_torch(tt_out_tt).float()

        pcc_pass, pcc_value = comp_pcc(ref_out, tt_out, pcc=0.99)
        import math

        scale = 1000.0
        half_dim = 160
        emb_freq = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_freq)
        sin_arg = scale * t_val * freqs
        sin_max = sin_arg.max().item()
        max_abs_err = (ref_out - tt_out).abs().max().item()
        print(f"{t_val:>10.4f} | {pcc_value:>10.6f} | {sin_max:>10.2f} | {max_abs_err:>12.4f}")

    print("=" * 72)
    # No assertion: this is a diagnostic; the assertion is in the caller
    # that uses these embeddings.
