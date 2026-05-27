# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for a full Qwen2 decoder layer used in CosyVoice LLM.

Tests the complete layer: RMSNorm → Attention → Residual → RMSNorm → MLP → Residual
with Qwen2-0.5B dimensions (hidden_size=896, 14 Q heads, 2 KV heads).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.cosy_voice.tt.ttnn_qwen2_decoder_layer import TtQwen2DecoderLayer
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


class PyTorchQwen2DecoderLayer(torch.nn.Module):
    """Simplified reference Qwen2 decoder layer for testing."""

    def __init__(self, hidden_size=896, num_heads=14, num_kv_heads=2, head_dim=64, intermediate_size=4864, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-attention norm
        self.input_layernorm = torch.nn.Module()
        self.input_layernorm_weight = torch.nn.Parameter(torch.ones(hidden_size))

        # Attention
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Post-attention norm
        self.post_attention_layernorm_weight = torch.nn.Parameter(torch.ones(hidden_size))

        # MLP
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

        self.eps = eps

    def _rms_norm(self, x, weight):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (weight * x).to(x.dtype)

    def forward(self, x):
        B, S, _ = x.shape
        residual = x

        # Pre-attention norm
        x = self._rms_norm(x, self.input_layernorm_weight)

        # Attention (simplified, no RoPE for unit test)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV for GQA
        kv_groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(kv_groups, dim=1)
        v = v.repeat_interleave(kv_groups, dim=1)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        attn = self.o_proj(attn)

        x = residual + attn

        # Post-attention norm + MLP
        residual = x
        x = self._rms_norm(x, self.post_attention_layernorm_weight)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = residual + x

        return x


@pytest.mark.parametrize(
    "seq_len, pcc",
    [
        (1, 0.997),
        (32, 0.996),
    ],
    ids=["decode_1", "seq_32"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_qwen2_decoder_layer(
    seq_len: int,
    pcc: float,
    device: ttnn.Device,
    reset_seeds,
):
    """Test TTNN Qwen2 decoder layer against PyTorch reference."""

    torch.manual_seed(42)
    hidden_size = 896
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    intermediate_size = 4864
    eps = 1e-6

    # 1. Create reference model and input
    reference_model = PyTorchQwen2DecoderLayer(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, eps)
    reference_model.eval()
    input_tensor = torch.randn(1, seq_len, hidden_size)

    # 2. Run PyTorch reference
    with torch.no_grad():
        reference_output = reference_model(input_tensor)

    # 3. Convert weights to TTNN format
    # Build a state dict matching the expected key format for TtQwen2DecoderLayer
    ref_sd = reference_model.state_dict()
    weights = {}

    # Attention weights
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        for suffix in ["weight", "bias"]:
            key = f"{proj_name}.{suffix}"
            if key in ref_sd:
                tensor = ref_sd[key].clone()
                if suffix == "weight" and len(tensor.shape) == 2:
                    tensor = tensor.T.contiguous()
                while len(tensor.shape) < 2:
                    tensor = tensor.unsqueeze(0)
                tt_key = f"self_attn.{key}"
                weights[tt_key] = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                weights[tt_key] = ttnn.to_device(weights[tt_key], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # MLP weights
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        key = f"{proj_name}.weight"
        if key in ref_sd:
            tensor = ref_sd[key].clone().T.contiguous()
            while len(tensor.shape) < 2:
                tensor = tensor.unsqueeze(0)
            tt_key = f"mlp.{key}"
            weights[tt_key] = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            weights[tt_key] = ttnn.to_device(weights[tt_key], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Norm weights
    for norm_name in ["input_layernorm_weight", "post_attention_layernorm_weight"]:
        if norm_name in ref_sd:
            tensor = ref_sd[norm_name].clone()
            while len(tensor.shape) < 2:
                tensor = tensor.unsqueeze(0)
            key = norm_name.replace("_weight", ".weight")
            weights[key] = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            weights[key] = ttnn.to_device(weights[key], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # 4. Create TTNN decoder layer
    configs = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "rms_norm_eps": eps,
        "rope_theta": 1000000.0,
        "max_seq_len": 2048,
        "dtype": {"activations": ttnn.bfloat16, "weights": ttnn.bfloat16},
    }

    tt_layer = TtQwen2DecoderLayer(device, configs, weights, layer_idx=0)

    # 5. Prepare TTNN input
    tt_input = input_tensor.view(1, 1, seq_len, hidden_size)
    tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 6. Run TTNN decoder layer
    tt_output, _ = tt_layer(tt_input)

    # 7. Compare
    tt_output_torch = ttnn.to_torch(tt_output).view(1, seq_len, hidden_size)

    logger.info(f"Decoder layer test: seq_len={seq_len}")
    logger.info(comp_allclose(reference_output, tt_output_torch))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output_torch, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning(f"Decoder layer PCC {output_pcc} below threshold {pcc}")
        assert does_pass, f"PCC value {output_pcc} is lower than {pcc}"
