# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Qwen2 MLP block.

The Qwen2 MLP uses a SiLU-gated architecture:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

Dimensions: 896 -> 4864 (gate/up) -> 896 (down)
"""

import torch

import ttnn


class TtQwen2MLP(torch.nn.Module):
    """TTNN implementation of the Qwen2 gated MLP.

    Architecture:
        gate = SiLU(gate_proj(x))    # (B, S, 896) -> (B, S, 4864)
        up = up_proj(x)               # (B, S, 896) -> (B, S, 4864)
        output = down_proj(gate * up)  # (B, S, 4864) -> (B, S, 896)
    """

    def __init__(
        self,
        device: ttnn.Device,
        configs: dict,
        weights: dict,
    ):
        super().__init__()
        self.device = device
        self.configs = configs

        # Load weight tensors (already transposed in preprocessing)
        self.gate_proj_weight = weights["mlp.gate_proj.weight"]
        self.up_proj_weight = weights["mlp.up_proj.weight"]
        self.down_proj_weight = weights["mlp.down_proj.weight"]

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass for Qwen2 MLP.

        Args:
            hidden_states: Input tensor (1, 1, seq_len, 896)

        Returns:
            Output tensor (1, 1, seq_len, 896)
        """
        # Gate projection + SiLU activation
        gate = ttnn.linear(hidden_states, self.gate_proj_weight)
        gate = ttnn.silu(gate)

        # Up projection
        up = ttnn.linear(hidden_states, self.up_proj_weight)

        # Element-wise multiply gate * up
        intermediate = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Down projection
        output = ttnn.linear(intermediate, self.down_proj_weight)
        ttnn.deallocate(intermediate)

        return output
