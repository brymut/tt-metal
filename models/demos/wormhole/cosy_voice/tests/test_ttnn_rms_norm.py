# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for RMSNorm operation used in Qwen2 decoder layers.

Tests that ttnn.rms_norm matches PyTorch reference output within
the expected PCC threshold for CosyVoice's Qwen2 backbone.

Qwen2 uses RMSNorm with eps=1e-6 at hidden_size=896.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


class PyTorchRMSNorm(torch.nn.Module):
    """Reference PyTorch RMSNorm implementation matching Qwen2."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


@pytest.mark.parametrize(
    "batch, seq_len, hidden_size, eps, pcc",
    [
        (1, 1, 896, 1e-6, 0.9999),  # Decode: single token
        (1, 32, 896, 1e-6, 0.9999),  # Short sequence
        (1, 128, 896, 1e-6, 0.9999),  # Medium sequence
        (1, 512, 896, 1e-6, 0.9998),  # Longer sequence
    ],
    ids=[
        "decode_single_token",
        "short_sequence_32",
        "medium_sequence_128",
        "longer_sequence_512",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rms_norm(
    batch: int,
    seq_len: int,
    hidden_size: int,
    eps: float,
    pcc: float,
    device: ttnn.Device,
    reset_seeds,
):
    """Test ttnn.rms_norm against PyTorch reference for Qwen2 dimensions."""

    # 1. Create reference model and input
    torch.manual_seed(42)
    reference_model = PyTorchRMSNorm(hidden_size, eps)
    input_tensor = torch.randn(batch, seq_len, hidden_size)

    # 2. Run PyTorch reference
    reference_output = reference_model(input_tensor)

    # 3. Prepare TTNN weight
    weight = reference_model.weight.data.unsqueeze(0)  # (1, hidden_size)
    tt_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.to_device(tt_weight, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # 4. Prepare TTNN input (reshape to 4D as required by TTNN)
    tt_input = input_tensor.view(1, 1, batch * seq_len, hidden_size)
    tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 5. Run TTNN RMSNorm
    tt_output = ttnn.rms_norm(tt_input, weight=tt_weight, epsilon=eps)

    # 6. Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch.view(batch, seq_len, hidden_size)

    # 7. Compare
    logger.info(f"RMSNorm test: batch={batch}, seq_len={seq_len}, hidden_size={hidden_size}")
    logger.info(comp_allclose(reference_output, tt_output_torch))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output_torch, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning(f"RMSNorm PCC {output_pcc} below threshold {pcc}")
        assert does_pass, f"PCC value {output_pcc} is lower than {pcc}"
