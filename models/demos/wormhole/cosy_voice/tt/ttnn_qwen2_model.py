# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the full Qwen2 model (24 decoder layers).

This wraps all 24 Qwen2DecoderLayers with the final RMSNorm and provides
the forward_one_step interface used by CosyVoice's autoregressive decode loop.
"""

from typing import Dict, List, Optional, Tuple

import torch

import ttnn
from models.demos.wormhole.cosy_voice.tt.preprocessing import load_qwen2_weights
from models.demos.wormhole.cosy_voice.tt.ttnn_qwen2_decoder_layer import TtQwen2DecoderLayer


class TtQwen2Model(torch.nn.Module):
    """TTNN implementation of the 24-layer Qwen2-0.5B transformer.

    Provides both full-sequence forward (for prefill) and single-step
    forward_one_step (for autoregressive decode with KV cache).
    """

    def __init__(
        self,
        device: ttnn.Device,
        configs: dict,
        state_dict: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.device = device
        self.configs = configs
        self.num_layers = configs["num_layers"]  # 24

        # Load the final RMSNorm
        final_norm_weight = state_dict.get("llm.model.norm.weight")
        if final_norm_weight is not None:
            while len(final_norm_weight.shape) < 2:
                final_norm_weight = final_norm_weight.unsqueeze(0)
            self.final_norm_weight = ttnn.from_torch(
                final_norm_weight, dtype=configs["dtype"]["weights"], layout=ttnn.TILE_LAYOUT
            )
            self.final_norm_weight = ttnn.to_device(
                self.final_norm_weight, device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            self.final_norm_weight = None

        # Build all decoder layers
        self.layers = []
        for layer_idx in range(self.num_layers):
            layer_weights = load_qwen2_weights(state_dict, device, layer_idx, dtype=configs["dtype"]["weights"])
            layer = TtQwen2DecoderLayer(device, configs, layer_weights, layer_idx)
            self.layers.append(layer)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
    ) -> Tuple[ttnn.Tensor, List[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Full forward pass through all 24 layers.

        Args:
            hidden_states: Input embeddings (1, 1, seq_len, 896)
            attention_mask: Causal mask
            position_ids: Position indices for RoPE
            kv_caches: List of KV caches per layer (None for first pass)

        Returns:
            Tuple of (output_hidden_states, new_kv_caches)
        """
        new_kv_caches = []

        for layer_idx, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[layer_idx] if kv_caches is not None else None

            hidden_states, new_kv_cache = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=layer_kv_cache,
            )
            new_kv_caches.append(new_kv_cache)

        # Final RMSNorm
        if self.final_norm_weight is not None:
            hidden_states = ttnn.rms_norm(
                hidden_states,
                weight=self.final_norm_weight,
                epsilon=self.configs["rms_norm_eps"],
            )

        return hidden_states, new_kv_caches

    def forward_one_step(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
    ) -> Tuple[ttnn.Tensor, List[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Single decode step with KV cache (autoregressive generation).

        This is the hot path for token generation — called once per token.
        Input is a single token embedding (1, 1, 1, 896).

        Args:
            hidden_states: Single-token input (1, 1, 1, 896)
            attention_mask: Full causal mask up to current position
            position_ids: Current position index
            kv_caches: KV caches from all previous steps

        Returns:
            Tuple of (output, updated_kv_caches)
        """
        return self.forward(hidden_states, attention_mask, position_ids, kv_caches)
