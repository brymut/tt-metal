# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Weight loading and preprocessing utilities for CosyVoice TTNN models.

Handles loading PyTorch weights from the reference CosyVoice model and
converting them to TTNN tensors with appropriate layouts and memory configs.
"""

from typing import Callable, Dict, Optional

import torch

import ttnn


class TtTensorLoader:
    """Loads PyTorch state_dict tensors and converts them to TTNN format.

    Follows the same pattern as Mamba's TtTensorLoader for consistency.
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        device: ttnn.Device,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.state_dict = state_dict
        self.device = device
        self.dtype = dtype

    def get_tensor_loader(self, prefix: str = "") -> Callable:
        """Returns a function that loads a named tensor from the state dict.

        Args:
            prefix: Key prefix for the current module (e.g., 'llm.llm.model.layers.0.')
        """

        def load_fn(
            name: str,
            *,
            dtype: Optional[ttnn.DataType] = None,
            layout: ttnn.Layout = ttnn.TILE_LAYOUT,
            memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
            transpose: bool = False,
        ) -> ttnn.Tensor:
            full_key = f"{prefix}{name}" if prefix else name
            if full_key not in self.state_dict:
                raise KeyError(
                    f"Key '{full_key}' not found in state dict. "
                    f"Available keys starting with '{prefix}': "
                    f"{[k for k in self.state_dict.keys() if k.startswith(prefix)][:10]}"
                )

            tensor = self.state_dict[full_key]

            if transpose:
                tensor = tensor.T.contiguous()

            # Ensure tensor is at least 2D for TILE_LAYOUT
            while len(tensor.shape) < 2:
                tensor = tensor.unsqueeze(0)

            use_dtype = dtype if dtype is not None else self.dtype

            tt_tensor = ttnn.from_torch(tensor, dtype=use_dtype, layout=layout)
            tt_tensor = ttnn.to_device(tt_tensor, self.device, memory_config=memory_config)
            return tt_tensor

        return load_fn


def load_qwen2_weights(
    state_dict: Dict[str, torch.Tensor],
    device: ttnn.Device,
    layer_idx: int,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> Dict[str, ttnn.Tensor]:
    """Load weights for a single Qwen2 decoder layer.

    Args:
        state_dict: Full model state dict (from llm.pt)
        device: TTNN device
        layer_idx: Transformer layer index (0-23)
        dtype: Target data type

    Returns:
        Dict mapping weight names to TTNN tensors
    """
    prefix = f"llm.model.layers.{layer_idx}."
    weights = {}

    weight_names = [
        # Attention
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        # MLP
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        # Norms
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]

    # Also check for bias terms
    bias_names = [
        "self_attn.q_proj.bias",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.bias",
    ]

    for name in weight_names:
        full_key = prefix + name
        if full_key in state_dict:
            tensor = state_dict[full_key]
            # Transpose weight matrices for linear layers (weight shape is [out, in])
            if "weight" in name and len(tensor.shape) == 2:
                tensor = tensor.T.contiguous()
            while len(tensor.shape) < 2:
                tensor = tensor.unsqueeze(0)
            weights[name] = ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            weights[name] = ttnn.to_device(weights[name], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    for name in bias_names:
        full_key = prefix + name
        if full_key in state_dict:
            tensor = state_dict[full_key]
            while len(tensor.shape) < 2:
                tensor = tensor.unsqueeze(0)
            weights[name] = ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            weights[name] = ttnn.to_device(weights[name], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return weights


def load_cosyvoice_llm_weights(
    state_dict: Dict[str, torch.Tensor],
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> Dict[str, ttnn.Tensor]:
    """Load the CosyVoice-specific LLM wrapper weights (embeddings, decoder head).

    These are the weights outside the Qwen2 backbone:
    - speech_embedding
    - llm_embedding
    - llm_decoder
    """
    weights = {}

    wrapper_weights = [
        "speech_embedding.weight",
        "llm_embedding.weight",
        "llm_decoder.weight",
        "llm_decoder.bias",
    ]

    for name in wrapper_weights:
        if name in state_dict:
            tensor = state_dict[name]
            if "decoder.weight" in name:
                tensor = tensor.T.contiguous()
            while len(tensor.shape) < 2:
                tensor = tensor.unsqueeze(0)
            weights[name] = ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            weights[name] = ttnn.to_device(weights[name], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return weights
