# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Qwen2 Grouped-Query Attention with Rotary Position Embeddings.

This module implements the attention mechanism used in each Qwen2 decoder layer:
- 14 query heads, 2 key/value heads (GQA ratio 7:1)
- Head dimension: 64
- Rotary positional embeddings (RoPE) with theta=1,000,000
- KV-cache for autoregressive generation

Uses HF-style ttnn.experimental.rotary_embedding following the Gemma4 pattern.
"""

from typing import Optional, Tuple

import torch

import ttnn


def precompute_rope_cos_sin_cache(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float = 1000000.0,
    device: ttnn.Device = None,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Precompute cos/sin caches for RoPE.

    Follows the standard HuggingFace-style RoPE computation:
        freq = 1 / (theta ^ (2i / dim))
        cos_cache[pos, i] = cos(pos * freq[i])
        sin_cache[pos, i] = sin(pos * freq[i])

    Args:
        head_dim: Dimension per attention head (64 for Qwen2-0.5B)
        max_seq_len: Maximum sequence length for cache pre-allocation
        rope_theta: Base frequency for RoPE (1,000,000 for Qwen2)
        device: TTNN device to place the caches on

    Returns:
        Tuple of (cos_cache, sin_cache) each of shape [1, 1, max_seq_len, head_dim]
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    # Outer product: (max_seq_len,) x (head_dim/2,) -> (max_seq_len, head_dim/2)
    freqs = torch.outer(positions, inv_freq)
    # Duplicate for full head_dim: [pos, d//2] -> [pos, d] via [cos, cos] interleave
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    cos_cache = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, max_seq_len, head_dim)
    sin_cache = emb.sin().unsqueeze(0).unsqueeze(0)

    tt_cos = ttnn.from_torch(cos_cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_sin = ttnn.from_torch(sin_cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    if device is not None:
        tt_cos = ttnn.to_device(tt_cos, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_sin = ttnn.to_device(tt_sin, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return tt_cos, tt_sin


def apply_rope(
    tensor: ttnn.Tensor,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    token_index: Optional[int] = None,
) -> ttnn.Tensor:
    """Apply HF-style rotary position embedding.

    Uses ttnn.experimental.rotary_embedding (not the llama variant).
    Follows the Gemma4 pattern for decode mode padding handling.

    Args:
        tensor: [1, heads, S, head_dim] (prefill) or padded for decode
        cos_cache: [1, 1, max_seq_len, head_dim]
        sin_cache: [1, 1, max_seq_len, head_dim]
        token_index: int for decode (position in sequence), None for prefill

    Returns:
        RoPE-applied tensor with same shape as input
    """
    orig_shape = tensor.shape
    result = ttnn.experimental.rotary_embedding(tensor, cos_cache, sin_cache, token_index)

    # In decode mode, dim 2 may get padded to TILE_HEIGHT (32).
    # Reshape and slice to restore original logical shape.
    if token_index is not None and result.shape[2] != orig_shape[2]:
        result = ttnn.reshape(
            result,
            (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]),
            (orig_shape[0], orig_shape[1], 32, orig_shape[3]),
        )
        result = result[:, :, : orig_shape[2]]

    return result


class TtQwen2Attention(torch.nn.Module):
    """TTNN implementation of Qwen2 GQA attention with RoPE and KV cache.

    Architecture:
        Input (B, S, 896) -> Q/K/V projections -> RoPE -> GQA SDPA -> Output projection
        Q: (B, 14, S, 64), K: (B, 2, S, 64), V: (B, 2, S, 64)
    """

    def __init__(
        self,
        device: ttnn.Device,
        configs: dict,
        weights: dict,
        layer_idx: int,
    ):
        super().__init__()
        self.device = device
        self.configs = configs
        self.layer_idx = layer_idx

        self.hidden_size = configs["hidden_size"]  # 896
        self.num_heads = configs["num_heads"]  # 14
        self.num_kv_heads = configs["num_kv_heads"]  # 2
        self.head_dim = configs["head_dim"]  # 64
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 7

        # Load weight tensors
        self.q_proj_weight = weights["self_attn.q_proj.weight"]
        self.k_proj_weight = weights["self_attn.k_proj.weight"]
        self.v_proj_weight = weights["self_attn.v_proj.weight"]
        self.o_proj_weight = weights["self_attn.o_proj.weight"]

        # Load bias tensors (Qwen2 attention has biases on Q/K/V)
        self.q_proj_bias = weights.get("self_attn.q_proj.bias")
        self.k_proj_bias = weights.get("self_attn.k_proj.bias")
        self.v_proj_bias = weights.get("self_attn.v_proj.bias")

        # Pre-compute RoPE cos/sin caches
        max_seq_len = configs.get("max_seq_len", 2048)
        rope_theta = configs.get("rope_theta", 1000000.0)
        self.cos_cache, self.sin_cache = precompute_rope_cos_sin_cache(self.head_dim, max_seq_len, rope_theta, device)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_ids: Optional[int] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Forward pass for Qwen2 GQA attention.

        Args:
            hidden_states: Input tensor (1, 1, seq_len, hidden_size)
            position_ids: Token index for decode mode (int), None for prefill
            attention_mask: Causal attention mask
            kv_cache: Tuple of (key_cache, value_cache) from previous steps

        Returns:
            Tuple of (output, new_kv_cache)
        """
        # Q/K/V projections
        query = ttnn.linear(hidden_states, self.q_proj_weight, bias=self.q_proj_bias)
        key = ttnn.linear(hidden_states, self.k_proj_weight, bias=self.k_proj_bias)
        value = ttnn.linear(hidden_states, self.v_proj_weight, bias=self.v_proj_bias)

        # Reshape for multi-head attention
        # query: (1, 1, S, 896) -> (1, 14, S, 64)
        # key:   (1, 1, S, 128) -> (1, 2, S, 64)
        # value: (1, 1, S, 128) -> (1, 2, S, 64)
        query = ttnn.reshape(query, (1, -1, self.num_heads, self.head_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.reshape(key, (1, -1, self.num_kv_heads, self.head_dim))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.reshape(value, (1, -1, self.num_kv_heads, self.head_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # Apply Rotary Position Embeddings (RoPE)
        query = apply_rope(query, self.cos_cache, self.sin_cache, position_ids)
        key = apply_rope(key, self.cos_cache, self.sin_cache, position_ids)

        # KV cache update
        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            key = ttnn.concat([key_cache, key], dim=2)
            value = ttnn.concat([value_cache, value], dim=2)
        new_kv_cache = (key, value)

        # Scaled dot-product attention with GQA
        # ttnn SDPA handles the GQA head expansion internally
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=True if attention_mask is None else False,
        )

        # Reshape back: (1, 14, S, 64) -> (1, 1, S, 896)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (1, 1, -1, self.hidden_size))

        # Output projection
        attn_output = ttnn.linear(attn_output, self.o_proj_weight)

        return attn_output, new_kv_cache
