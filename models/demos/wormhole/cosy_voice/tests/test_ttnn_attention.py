# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for Qwen2 attention block (GQA + RoPE) used in CosyVoice LLM.

Tests the full attention path:
1. Q/K/V linear projections
2. Multi-head reshape
3. Rotary positional embeddings (RoPE)
4. Grouped-query attention (SDPA)
5. Output projection

Qwen2-0.5B: 14 query heads, 2 KV heads, head_dim=64, hidden_size=896
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.cosy_voice.tt.ttnn_qwen2_attention import TtQwen2Attention
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


class PyTorchQwen2Attention(torch.nn.Module):
    """Simplified reference Qwen2 attention for testing."""

    def __init__(self, hidden_size=896, num_heads=14, num_kv_heads=2, head_dim=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads for GQA
        num_kv_groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

        # SDPA with causal mask
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        return self.o_proj(attn_output)


@pytest.mark.parametrize(
    "seq_len, pcc",
    [
        (1, 0.998),
        (32, 0.997),
        (128, 0.996),
    ],
    ids=["decode_1", "seq_32", "seq_128"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_qwen2_attention(
    seq_len: int,
    pcc: float,
    device: ttnn.Device,
    reset_seeds,
):
    """Test TTNN Qwen2 attention against PyTorch reference."""

    torch.manual_seed(42)
    hidden_size = 896
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64

    # 1. Create reference model and input
    reference_model = PyTorchQwen2Attention(hidden_size, num_heads, num_kv_heads, head_dim)
    reference_model.eval()
    input_tensor = torch.randn(1, seq_len, hidden_size)

    # 2. Run PyTorch reference
    with torch.no_grad():
        reference_output = reference_model(input_tensor)

    # 3. Extract weights into TTNN format
    state_dict = reference_model.state_dict()
    weights = {}
    for name, param in state_dict.items():
        tensor = param.clone()
        # Transpose weight matrices (nn.Linear stores as [out, in])
        if "weight" in name and len(tensor.shape) == 2:
            tensor = tensor.T.contiguous()
        while len(tensor.shape) < 2:
            tensor = tensor.unsqueeze(0)
        weights[name] = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        weights[name] = ttnn.to_device(weights[name], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Rename keys to match TtQwen2Attention expectations
    renamed_weights = {}
    for k, v in weights.items():
        renamed_weights[f"self_attn.{k}"] = v

    # 4. Create TTNN attention module
    configs = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "dtype": {"activations": ttnn.bfloat16, "weights": ttnn.bfloat16},
    }

    tt_attention = TtQwen2Attention(device, configs, renamed_weights, layer_idx=0)

    # 5. Prepare TTNN input
    tt_input = input_tensor.view(1, 1, seq_len, hidden_size)
    tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 6. Run TTNN attention
    tt_output, _ = tt_attention(tt_input)

    # 7. Compare
    tt_output_torch = ttnn.to_torch(tt_output).view(1, seq_len, hidden_size)

    logger.info(f"Attention test: seq_len={seq_len}")
    logger.info(comp_allclose(reference_output, tt_output_torch))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output_torch, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning(f"Attention PCC {output_pcc} below threshold {pcc}")
        assert does_pass, f"PCC value {output_pcc} is lower than {pcc}"
