# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of a single Qwen2 decoder layer.

Each layer consists of:
1. RMSNorm -> Self-Attention (GQA + RoPE) -> Residual add
2. RMSNorm -> MLP (SiLU-gated) -> Residual add
"""

from typing import Optional, Tuple

import torch

import ttnn
from models.demos.wormhole.cosy_voice.tt.ttnn_qwen2_attention import TtQwen2Attention
from models.demos.wormhole.cosy_voice.tt.ttnn_qwen2_mlp import TtQwen2MLP


class TtQwen2DecoderLayer(torch.nn.Module):
    """TTNN implementation of a single Qwen2 transformer decoder layer.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
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
        self.rms_norm_eps = configs["rms_norm_eps"]

        # RMSNorm weights
        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]

        # Sub-modules
        self.self_attn = TtQwen2Attention(device, configs, weights, layer_idx)
        self.mlp = TtQwen2MLP(device, configs, weights)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Forward pass for a single Qwen2 decoder layer.

        Args:
            hidden_states: Input tensor (1, 1, seq_len, 896)
            position_ids: Position indices for RoPE
            attention_mask: Causal attention mask
            kv_cache: KV cache from previous decode steps

        Returns:
            Tuple of (output_hidden_states, new_kv_cache)
        """
        # Store residual
        residual = hidden_states

        # Pre-attention RMSNorm
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=self.input_layernorm_weight,
            epsilon=self.rms_norm_eps,
        )

        # Self-attention
        attn_output, new_kv_cache = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        ttnn.deallocate(hidden_states)

        # Residual connection
        hidden_states = ttnn.add(residual, attn_output)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_output)

        # Store residual for MLP
        residual = hidden_states

        # Post-attention RMSNorm
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=self.post_attention_layernorm_weight,
            epsilon=self.rms_norm_eps,
        )

        # MLP
        mlp_output = self.mlp(hidden_states)
        ttnn.deallocate(hidden_states)

        # Residual connection
        hidden_states = ttnn.add(residual, mlp_output)
        ttnn.deallocate(residual)
        ttnn.deallocate(mlp_output)

        return hidden_states, new_kv_cache
