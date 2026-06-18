# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.wormhole.cosyvoice.tt.attention import TtRelPositionMultiHeadedAttention


class TtPositionwiseFeedForward(torch.nn.Module):
    def __init__(self, device, state_dict, base_address, idim, hidden_units, activation_name="relu", dropout_rate=0.0):
        super().__init__()
        self.device = device
        self.idim = idim
        self.hidden_units = hidden_units

        self.w1_weight = state_dict[f"{base_address}.w_1.weight"]
        self.w1_bias = state_dict[f"{base_address}.w_1.bias"]
        self.w2_weight = state_dict[f"{base_address}.w_2.weight"]
        self.w2_bias = state_dict[f"{base_address}.w_2.bias"]

        self.tt_w1_weight = ttnn.from_torch(
            self.w1_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_w1_bias = ttnn.from_torch(self.w1_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_w2_weight = ttnn.from_torch(
            self.w2_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_w2_bias = ttnn.from_torch(self.w2_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        self.activation_name = activation_name
        if activation_name == "gelu":
            self.activation = "gelu_approx"
        elif activation_name == "swish":
            self.activation = "silu"
        else:
            self.activation = "relu"

    def forward(self, x):
        x = ttnn.linear(
            x,
            self.tt_w1_weight,
            bias=self.tt_w1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            activation=self.activation,
        )
        x = ttnn.linear(
            x, self.tt_w2_weight, bias=self.tt_w2_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        return x


class TtTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, device, config, base_address, size, state_dict=None, num_layers=14):
        super().__init__()
        self.device = device
        self.config = config
        self.size = size
        self.normalize_before = True  # CosyVoice uses normalize_before=True

        self.attn = TtRelPositionMultiHeadedAttention(
            device,
            state_dict,
            f"{base_address}.self_attn",
            config["attention_heads"],
            size,
            config.get("attention_dropout_rate", 0.0),
            config.get("key_bias", True),
        )

        activation_type = config.get("activation_type", "relu")
        self.feed_forward = TtPositionwiseFeedForward(
            device,
            state_dict,
            f"{base_address}.feed_forward",
            size,
            config["linear_units"],
            activation_type,
            config["dropout_rate"],
        )

    def forward(
        self, x, mask, pos_emb, mask_pad=None, att_cache=torch.zeros((0, 0, 0, 0)), cnn_cache=torch.zeros((0, 0, 0, 0))
    ):
        residual = x

        if self.normalize_before:
            x_norm = self._layer_norm(x)
        else:
            x_norm = x

        x_att, new_att_cache = self.attn(x_norm, x_norm, x_norm, mask, pos_emb=pos_emb, cache=att_cache)

        if not self.normalize_before:
            x_att = self._layer_norm(x_att)

        x = ttnn.add(residual, x_att, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        residual = x

        if self.normalize_before:
            x_norm = self._layer_norm(x)
        else:
            x_norm = x

        x_ff = self.feed_forward(x_norm)

        if not self.normalize_before:
            x_ff = self._layer_norm(x_ff)

        x = ttnn.add(residual, x_ff, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        return x

    def _layer_norm(self, x):
        x_torch = ttnn.to_torch(x)
        x_norm = torch.nn.functional.layer_norm(x_torch, x_torch.shape[-1:], eps=1e-12)
        return ttnn.from_torch(x_norm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)


class TtBaseEncoder(torch.nn.Module):
    def __init__(self, device, config, base_address, num_layers, state_dict=None):
        super().__init__()
        self.device = device
        self.config = config
        self.base_address = base_address
        self.num_layers = num_layers
        self.after_norm = None
        self.encoders = torch.nn.ModuleList()

    def output_size(self) -> int:
        return self.config.get("output_size", self.config.get("hidden_size", 256))

    def build(self, state_dict):
        for i in range(self.num_layers):
            layer = TtTransformerEncoderLayer(
                self.device, self.config, f"{self.base_address}.{i}", self.output_size(), state_dict=state_dict
            )
            self.encoders.append(layer)

    def forward(self, xs, xs_lens, decoding_chunk_size=0, num_decoding_left_chunks=-1):
        T = xs.shape[1]

        masks = torch.zeros(xs.shape[0], 1, T, device="cpu", dtype=torch.bool)
        for b in range(xs.shape[0]):
            masks[b, :, xs_lens[b] :] = True

        pos_emb = self._get_positional_encoding(T)

        for layer in self.encoders:
            xs = layer(xs, masks, pos_emb)

        return xs, masks

    def _get_positional_encoding(self, size):
        d_model = self.config.get("output_size", self.config.get("hidden_size", 256))
        pe = torch.zeros(1, size, d_model)
        position = torch.arange(0, size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward_chunk(
        self,
        xs,
        offset,
        required_cache_size,
        att_cache=torch.zeros((0, 0, 0, 0)),
        cnn_cache=torch.zeros((0, 0, 0, 0)),
        att_mask=torch.ones((0, 0, 0), dtype=torch.bool),
    ):
        tmp_masks = torch.ones(1, xs.shape[1], device="cpu", dtype=torch.bool).unsqueeze(1)
        pos_emb = self._get_positional_encoding(xs.shape[1])

        r_att_cache = []
        r_cnn_cache = []

        for i, layer in enumerate(self.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs, tmp_masks, pos_emb, att_cache=att_cache[i : i + 1], cnn_cache=cnn_cache[i]
            )
            r_att_cache.append(new_att_cache)
            r_cnn_cache.append(new_cnn_cache)

        return xs, torch.cat(r_att_cache, dim=0), torch.cat(r_cnn_cache, dim=0)


class TtTransformerEncoder(TtBaseEncoder):
    def __init__(self, device, config, base_address, state_dict):
        super().__init__(device, config, base_address, config["num_blocks"], state_dict)
        self.build(state_dict)

    def output_size(self) -> int:
        return self.config.get("output_size", self.config.get("hidden_size", 256))
