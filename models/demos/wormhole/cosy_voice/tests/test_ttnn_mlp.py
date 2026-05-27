# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for Qwen2 MLP block used in CosyVoice LLM.

Tests the SiLU-gated MLP:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

Qwen2-0.5B dimensions: 896 -> 4864 -> 896
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.cosy_voice.tt.ttnn_qwen2_mlp import TtQwen2MLP
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


class PyTorchQwen2MLP(torch.nn.Module):
    """Reference PyTorch Qwen2 MLP."""

    def __init__(self, hidden_size=896, intermediate_size=4864):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


@pytest.mark.parametrize(
    "seq_len, pcc",
    [
        (1, 0.999),
        (32, 0.998),
        (128, 0.997),
    ],
    ids=["decode_1", "seq_32", "seq_128"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_qwen2_mlp(
    seq_len: int,
    pcc: float,
    device: ttnn.Device,
    reset_seeds,
):
    """Test TTNN Qwen2 MLP against PyTorch reference."""

    torch.manual_seed(42)
    hidden_size = 896
    intermediate_size = 4864

    # 1. Create reference model and input
    reference_model = PyTorchQwen2MLP(hidden_size, intermediate_size)
    reference_model.eval()
    input_tensor = torch.randn(1, seq_len, hidden_size)

    # 2. Run PyTorch reference
    with torch.no_grad():
        reference_output = reference_model(input_tensor)

    # 3. Extract weights into TTNN format
    weights = {}
    for name, param in reference_model.state_dict().items():
        tensor = param.clone()
        if len(tensor.shape) == 2:
            tensor = tensor.T.contiguous()
        while len(tensor.shape) < 2:
            tensor = tensor.unsqueeze(0)
        weights[f"mlp.{name}"] = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        weights[f"mlp.{name}"] = ttnn.to_device(weights[f"mlp.{name}"], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # 4. Create TTNN MLP module
    configs = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": {"activations": ttnn.bfloat16, "weights": ttnn.bfloat16},
    }

    tt_mlp = TtQwen2MLP(device, configs, weights)

    # 5. Prepare TTNN input
    tt_input = input_tensor.view(1, 1, seq_len, hidden_size)
    tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 6. Run TTNN MLP
    tt_output = tt_mlp(tt_input)

    # 7. Compare
    tt_output_torch = ttnn.to_torch(tt_output).view(1, seq_len, hidden_size)

    logger.info(f"MLP test: seq_len={seq_len}")
    logger.info(comp_allclose(reference_output, tt_output_torch))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output_torch, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning(f"MLP PCC {output_pcc} below threshold {pcc}")
        assert does_pass, f"PCC value {output_pcc} is lower than {pcc}"
