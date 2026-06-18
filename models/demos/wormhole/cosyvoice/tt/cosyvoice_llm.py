# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

cosyvoice_path = Path(__file__).parent.parent / "reference" / "CosyVoice"
if str(cosyvoice_path) not in sys.path:
    sys.path.insert(0, str(cosyvoice_path))

from cosyvoice.utils.common import IGNORE_ID
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from models.demos.wormhole.cosyvoice.tt.attention import TtRelPositionMultiHeadedAttention

# Module-level compute kernel config: HiFi4 for higher precision math.
# This prevents greedy token divergence after step 1 in the LLM and
# prevents regression when moving attention to device.
# See HANDOFF §3 Session 2 and Session 8.
_LLM_LINEAR_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
)


class TtEspnetRelPositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        # Match the reference EspnetRelPositionalEncoding: pre-build the full
        # `pe` buffer to `max_len` once in __init__ (reference does
        # `self.extend_pe(torch.tensor(0.0).expand(1, max_len))`). The buffer is
        # never shrunk afterwards; `position_encoding` just slices into it.
        # Pre-building is what makes the streaming `forward_chunk` path correct:
        # `position_encoding(offset=0, size=total_len)` returns a full
        # `2*total_len-1`-column slice for any total_len <= max_len, which is
        # exactly what the reference's attention rel_shift expects.
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.max_len = max_len
        self.device = torch.device("cpu")
        self.pe = None
        self._built_size = 0
        self._build_pe(max_len)

    def _build_pe(self, size):
        pe_positive = torch.zeros(size, self.d_model, device=self.device)
        pe_negative = torch.zeros(size, self.d_model, device=self.device)
        position = torch.arange(0, size, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=self.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self.pe = torch.cat([pe_positive, pe_negative], dim=1)
        self._built_size = size

    def _ensure_pe(self, needed_size):
        if self.pe is None or self._built_size < needed_size:
            self._build_pe(needed_size)

    def position_encoding(self, offset, size):
        total = size + offset
        self._ensure_pe(total)
        if self.pe.size(1) // 2 < total:
            self._build_pe(total)
        start = self.pe.size(1) // 2 - size - offset + 1
        end = self.pe.size(1) // 2 + size + offset
        return self.pe[:, start:end]

    def forward(self, x, offset=0):
        T = x.size(1)
        self._ensure_pe(T + offset)
        x = x * self.xscale
        pe = self.position_encoding(offset=offset, size=T)
        return x, pe


class TtPositionwiseFeedForward(nn.Module):
    def __init__(self, device, state_dict, base_address, idim, hidden_units, dropout_rate=0.0, activation_name="relu"):
        super().__init__()
        self.device = device

        w1_w = state_dict[f"{base_address}.w_1.weight"]
        w1_b = state_dict[f"{base_address}.w_1.bias"]
        w2_w = state_dict[f"{base_address}.w_2.weight"]
        w2_b = state_dict[f"{base_address}.w_2.bias"]

        self.tt_w1 = ttnn.from_torch(w1_w.T, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_w1_b = ttnn.from_torch(w1_b, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_w2 = ttnn.from_torch(w2_w.T, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_w2_b = ttnn.from_torch(w2_b, layout=ttnn.TILE_LAYOUT, device=device)

        self.activation = activation_name

    def forward(self, x):
        if self.activation == "swish":
            act = "silu"
        elif self.activation == "gelu":
            act = "gelu_approx"
        else:
            act = "relu"
        # Cast input to fp32 so the FFN's 1024→4096→1024 chain runs in fp32.
        # The 4096-dim intermediate (after w1+ReLU) is the largest tensor in the
        # LLM body; keeping it in bf16 loses 8 mantissa bits per element and
        # is a major source of the LLM's per-step error accumulation.
        x_fp32 = ttnn.typecast(x, dtype=ttnn.float32)
        x_fp32 = ttnn.linear(
            x_fp32,
            self.tt_w1,
            bias=self.tt_w1_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            activation=act,
            compute_kernel_config=_LLM_LINEAR_KERNEL_CFG,
        )
        x_fp32 = ttnn.linear(
            x_fp32,
            self.tt_w2,
            bias=self.tt_w2_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=_LLM_LINEAR_KERNEL_CFG,
        )
        # Return fp32 — the encoder layer's residual add is in fp32.
        return x_fp32


class TtTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        size,
        self_attn,
        feed_forward,
        dropout_rate=0.0,
        normalize_before=True,
        norm1_key="norm1",
        norm2_key="norm2",
    ):
        super().__init__()
        self.device = device
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = dropout_rate
        self.size = size
        self.normalize_before = normalize_before

        self.norm1_weight = state_dict[f"{base_address}.{norm1_key}.weight"]
        self.norm1_bias = state_dict[f"{base_address}.{norm1_key}.bias"]
        self.norm2_weight = state_dict[f"{base_address}.{norm2_key}.weight"]
        self.norm2_bias = state_dict[f"{base_address}.{norm2_key}.bias"]

        # Pre-upload norm weights/biases to device for ttnn.layer_norm
        # (was previously: torch.f.layer_norm on host with explicit upload
        # round-trip per call — 2 syncs per layer per step)
        self.tt_norm1_weight = ttnn.from_torch(
            self.norm1_weight,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.tt_norm1_bias = ttnn.from_torch(
            self.norm1_bias,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.tt_norm2_weight = ttnn.from_torch(
            self.norm2_weight,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.tt_norm2_bias = ttnn.from_torch(
            self.norm2_bias,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def _layer_norm(self, x, weight, bias):
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            epsilon=1e-12,
        )

    def forward(self, x, mask=None, pos_emb=None):
        if self.normalize_before:
            x_norm = self._layer_norm(x, self.tt_norm1_weight, self.tt_norm1_bias)
        else:
            x_norm = x

        out, _ = self.self_attn(x_norm, x_norm, x_norm, mask, pos_emb=pos_emb)

        x = ttnn.add(x, out, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        if not self.normalize_before:
            x = self._layer_norm(x, self.tt_norm1_weight, self.tt_norm1_bias)

        if self.normalize_before:
            x_norm = self._layer_norm(x, self.tt_norm2_weight, self.tt_norm2_bias)
        else:
            x_norm = x

        out = self.feed_forward(x_norm)

        x = ttnn.add(x, out, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        if not self.normalize_before:
            x = self._layer_norm(x, self.tt_norm2_weight, self.tt_norm2_bias)

        return x, mask, None, None

    def forward_chunk(self, x, att_mask, pos_emb, att_cache=None):
        # Typecast residual to fp32 — the attention output and FFN output are
        # both fp32 (see attention.py and TtPositionwiseFeedForward), so the
        # residual add stays in fp32 throughout the encoder. This eliminates
        # the 8-bit bf16 quantization in the residual stream that compounds
        # over 14 layers and is a major source of LLM token divergence.
        x = ttnn.typecast(x, dtype=ttnn.float32)
        residual = x
        if self.normalize_before:
            x_norm = self._layer_norm(x, self.tt_norm1_weight, self.tt_norm1_bias)
        else:
            x_norm = x

        # Typecast back to bf16 for the attention's Q/K/V linear (bf16 weight).
        # The Q/K/V projection is bf16-precision regardless of input dtype.
        x_norm_attn = ttnn.typecast(x_norm, dtype=ttnn.bfloat16)
        out, new_att_cache = self.self_attn(
            x_norm_attn, x_norm_attn, x_norm_attn, att_mask, pos_emb=pos_emb, cache=att_cache
        )

        x = ttnn.add(residual, out, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)

        if not self.normalize_before:
            x = self._layer_norm(x, self.tt_norm1_weight, self.tt_norm1_bias)

        residual = x
        if self.normalize_before:
            x_norm = self._layer_norm(x, self.tt_norm2_weight, self.tt_norm2_bias)
        else:
            x_norm = x

        # FFN typecasts input to fp32 internally and returns fp32, so no
        # typecast needed here.
        out = self.feed_forward(x_norm)

        x = ttnn.add(residual, out, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)

        if not self.normalize_before:
            x = self._layer_norm(x, self.tt_norm2_weight, self.tt_norm2_bias)

        # Typecast back to bf16 for the next layer's input (and for the encoder's
        # final layer_norm and the llm_decoder downstream).
        x = ttnn.typecast(x, dtype=ttnn.bfloat16)

        return x, new_att_cache


class TtLinearNoSubsampling(nn.Module):
    def __init__(
        self, device, state_dict, base_address, idim, odim, dropout_rate=0.0, pos_enc_class=None, use_relu=False
    ):
        super().__init__()
        self.device = device

        w0_w = state_dict[f"{base_address}.out.0.weight"]
        w0_b = state_dict[f"{base_address}.out.0.bias"]
        w1_w = state_dict[f"{base_address}.out.1.weight"]
        w1_b = state_dict[f"{base_address}.out.1.bias"]

        self.tt_w0 = ttnn.from_torch(w0_w.T, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_w0_b = ttnn.from_torch(w0_b, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_w1 = ttnn.from_torch(w1_w, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_w1_b = ttnn.from_torch(w1_b, layout=ttnn.TILE_LAYOUT, device=device)

        self.pos_enc = pos_enc_class
        self.use_relu = use_relu

    def forward(self, x, mask, offset=0):
        x = ttnn.linear(
            x,
            self.tt_w0,
            bias=self.tt_w0_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_LLM_LINEAR_KERNEL_CFG,
        )

        # LayerNorm must come BEFORE ReLU to match the reference
        # (Linear -> LayerNorm -> Dropout -> ReLU)
        x = ttnn.layer_norm(x, weight=self.tt_w1, bias=self.tt_w1_b, memory_config=ttnn.L1_MEMORY_CONFIG, epsilon=1e-05)

        if self.use_relu:
            x_torch = ttnn.to_torch(x)
            x_torch = torch.relu(x_torch)
            x = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, device=self.device)

        if self.pos_enc is not None:
            x_torch = ttnn.to_torch(x)
            x_torch, pos_emb = self.pos_enc.forward(x_torch)
            x = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, device=self.device)
            return x, pos_emb, mask

        return x, mask


class TtTransformerEncoder(nn.Module):
    def __init__(self, device, state_dict, config):
        super().__init__()
        self.device = device
        self.config = config
        self.num_blocks = config["num_blocks"]
        self.size = config["output_size"]
        self.encoder_prefix = config["encoder_prefix"]
        norm1_key = config.get("norm1_key", "norm1")
        norm2_key = config.get("norm2_key", "norm2")

        self.embed = TtLinearNoSubsampling(
            device,
            state_dict,
            f"{config['encoder_prefix']}.embed",
            config["input_size"],
            config["output_size"],
            config.get("dropout_rate", 0.0),
            TtEspnetRelPositionalEncoding(config["output_size"]),
            use_relu=config.get("use_relu", False),
        )

        self.encoders = nn.ModuleList()
        for i in range(self.num_blocks):
            attn_base = f"{config['encoder_prefix']}.encoders.{i}.self_attn"
            attn = TtRelPositionMultiHeadedAttention(
                device, state_dict, attn_base, config["attention_heads"], self.size
            )
            act = config.get("activation_type", "relu")
            ff_base = f"{config['encoder_prefix']}.encoders.{i}.feed_forward"
            ff = TtPositionwiseFeedForward(
                device, state_dict, ff_base, self.size, config["linear_units"], activation_name=act
            )
            layer_base = f"{config['encoder_prefix']}.encoders.{i}"
            layer = TtTransformerEncoderLayer(
                device,
                state_dict,
                layer_base,
                self.size,
                attn,
                ff,
                config["dropout_rate"],
                normalize_before=True,
                norm1_key=norm1_key,
                norm2_key=norm2_key,
            )
            self.encoders.append(layer)

        after_norm_w = state_dict[f"{config['encoder_prefix']}.after_norm.weight"]
        after_norm_b = state_dict[f"{config['encoder_prefix']}.after_norm.bias"]
        self.tt_after_norm_w = ttnn.from_torch(after_norm_w, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_after_norm_b = ttnn.from_torch(after_norm_b, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(self, x, mask=None):
        x, pos_emb, mask = self.embed(x, mask)

        for layer in self.encoders:
            x, mask, _, _ = layer(x, mask, pos_emb)

        x = ttnn.layer_norm(
            x,
            weight=self.tt_after_norm_w,
            bias=self.tt_after_norm_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            epsilon=1e-05,
        )

        return x, mask

    def forward_chunk(self, x_tt, offset, att_cache=None, att_mask=None):
        chunk_size = x_tt.shape[1]
        total_len = offset + chunk_size
        chunk_mask = torch.ones(1, 1, chunk_size, dtype=torch.bool)
        x_tt, pos_emb, _ = self.embed(x_tt, chunk_mask)
        # Mirror the reference BaseEncoder.forward_chunk: the pos_emb returned by
        # `embed` is computed for `chunk_size` only and is WRONG for the streaming
        # case (chunk_size=1, offset>0). The reference overrides it with
        # `position_encoding(offset - cache_t1, attention_key_size)` where
        # `attention_key_size = cache_t1 + chunk_size = total_len`. In the
        # CosyVoice LLM inference loop `offset == cache_t1` always holds (the
        # full history is cached, required_cache_size=-1), so the effective call
        # is `position_encoding(0, total_len)` -> a `2*total_len-1`-column slice
        # covering relative positions [-(total_len-1) .. total_len-1]. Without
        # this the streaming attention's matrix_bd collapses to a single column
        # and the LLM diverges from the reference after the first token.
        cache_t1 = 0
        if att_cache is not None and att_cache.numel() > 0:
            cache_t1 = att_cache.size(2)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.pos_enc.position_encoding(offset=offset - cache_t1, size=attention_key_size)
        if pos_emb.dtype != torch.float32:
            pos_emb = pos_emb.float()

        new_caches = []
        for i, layer in enumerate(self.encoders):
            layer_cache = None
            if att_cache is not None and att_cache.size(0) > 0:
                layer_cache = att_cache[i : i + 1]
            x_tt, new_cache = layer.forward_chunk(x_tt, att_mask, pos_emb, att_cache=layer_cache)
            new_caches.append(new_cache)

        if new_caches and new_caches[0].numel() > 0:
            new_att_cache = torch.cat(new_caches, dim=0)
        else:
            new_att_cache = torch.zeros(0, 0, 0, 0)

        x_tt = ttnn.layer_norm(
            x_tt,
            weight=self.tt_after_norm_w,
            bias=self.tt_after_norm_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            epsilon=1e-05,
        )

        return x_tt, new_att_cache


class TtCosyVoiceLLM(torch.nn.Module):
    def __init__(self, device, config, args, state_dict=None):
        super().__init__()
        self.device = device
        self.config = config
        self.args = args

        self.text_embedding = nn.Embedding(51866, 512)
        self.text_encoder_affine_layer = nn.Linear(1024, 1024)
        self.llm_embedding = nn.Embedding(2, 1024)
        self.speech_embedding = nn.Embedding(4096, 1024)
        self.spk_embed_affine_layer = nn.Linear(192, 1024)
        self.llm_decoder = nn.Linear(1024, 4097)

        if state_dict is not None:
            self.text_embedding.load_state_dict({"weight": state_dict["text_embedding.weight"]})
            self.text_encoder_affine_layer.load_state_dict(
                {
                    "weight": state_dict["text_encoder_affine_layer.weight"],
                    "bias": state_dict["text_encoder_affine_layer.bias"],
                }
            )
            self.llm_embedding.load_state_dict({"weight": state_dict["llm_embedding.weight"]})
            self.speech_embedding.load_state_dict({"weight": state_dict["speech_embedding.weight"]})
            self.spk_embed_affine_layer.load_state_dict(
                {
                    "weight": state_dict["spk_embed_affine_layer.weight"],
                    "bias": state_dict["spk_embed_affine_layer.bias"],
                }
            )
            self.llm_decoder.load_state_dict(
                {"weight": state_dict["llm_decoder.weight"], "bias": state_dict["llm_decoder.bias"]}
            )

        llm_config = {
            "num_blocks": 14,
            "attention_heads": 16,
            "output_size": 1024,
            "input_size": 1024,
            "linear_units": 4096,
            "dropout_rate": 0.1,
            "activation_type": "relu",
            "use_relu": True,
            "encoder_prefix": "llm",
            "norm1_key": "norm1",
            "norm2_key": "norm2",
        }
        encoder_config = {
            "num_blocks": 6,
            "attention_heads": 16,
            "output_size": 1024,
            "input_size": 512,
            "linear_units": 4096,
            "dropout_rate": 0.1,
            "activation_type": "swish",
            "use_relu": False,
            "encoder_prefix": "text_encoder",
            "norm1_key": "norm_mha",
            "norm2_key": "norm_ff",
        }

        self.llm_encoder = TtTransformerEncoder(device, state_dict, llm_config)
        self.text_encoder = TtTransformerEncoder(device, state_dict, encoder_config)

        self.sos = 0
        self.task_id = 1
        self.speech_token_size = 4096
        self.eos_token = self.speech_token_size
        self._default_sampling = 25

    def _sampling_ids(self, weighted_scores, decoded_tokens, sampling, ignore_eos=True):
        """Sample the next speech token.

        Mirrors the reference ``ras_sampling`` (Repetition Aware Sampling
        from VALL-E 2 / utils/common.py:138-144): first do nucleus
        sampling (top-p + top-k), then if the chosen token has been
        repeated in the last ``win_size=10`` tokens with frequency
        ``>= win_size * tau_r = 1.0`` (i.e. any repetition at all),
        set its score to -inf and fall back to ``random_sampling``
        (random multinomial from the full distribution).

        Without RAS the TT LLM collapses to a degenerate token (e.g.
        193, 193, 193, ...) when its bf16-noisy logits put a non-argmax
        token at the top, because the reference would have detected
        the loop and broken out but the TT sampler wouldn't.

        Sampling modes:
          * ``sampling <= 0``: pure argmax (greedy, no RAS)
          * ``1 <= sampling <= 25``: top-k multinomial
          * ``sampling > 25`` (e.g. 25 == 4097): full multinomial
        """
        scores = weighted_scores
        if ignore_eos:
            scores = scores.clone()
            scores[self.eos_token] = -float("inf")

        if sampling is None or sampling <= 0:
            return int(scores.argmax().item())

        # Nucleus sampling (top-p + top-k) - matches reference defaults
        top_p = 0.8
        top_k = min(sampling, scores.size(-1))
        probs_all = scores.softmax(dim=-1)
        sorted_probs, sorted_idx = probs_all.sort(descending=True, stable=True)
        cum_prob = 0.0
        nucleus_probs = []
        nucleus_indices = []
        for i in range(len(sorted_idx)):
            if cum_prob < top_p and len(nucleus_probs) < top_k:
                cum_prob += sorted_probs[i].item()
                nucleus_probs.append(sorted_probs[i])
                nucleus_indices.append(sorted_idx[i])
            else:
                break
        if not nucleus_probs:
            return int(scores.argmax().item())
        nucleus_probs_t = torch.stack(nucleus_probs)
        # Renormalize over the nucleus
        nucleus_probs_t = nucleus_probs_t / nucleus_probs_t.sum()
        nucleus_idx = int(nucleus_probs_t.multinomial(1, replacement=True).item())
        top_ids = int(nucleus_indices[nucleus_idx].item())

        # Repetition-aware penalty: if this token appeared >= 1 time in
        # the last win_size=10 decoded tokens, reject and resample.
        win_size = 10
        tau_r = 0.1
        if len(decoded_tokens) > 0:
            recent = decoded_tokens[-win_size:]
            rep_num = sum(1 for t in recent if t == top_ids)
            if rep_num >= win_size * tau_r:  # 1.0
                # Penalize and fall back to a fully-random sample
                scores[top_ids] = -float("inf")
                # random_sampling: sample from full softmax distribution
                top_ids = int(scores.softmax(dim=-1).multinomial(1, replacement=True).item())
        return top_ids

    def _encode_text(self, text_token):
        text_emb = self.text_embedding(text_token)
        text_tt = ttnn.from_torch(text_emb, layout=ttnn.TILE_LAYOUT, device=self.device)
        mask = torch.ones(1, 1, text_emb.shape[1], dtype=torch.bool)
        text_encoded, _ = self.text_encoder(text_tt, mask)
        text_encoded = ttnn.to_torch(text_encoded).float()
        return self.text_encoder_affine_layer(text_encoded)

    @torch.inference_mode()
    def inference(
        self,
        text,
        text_len,
        prompt_text,
        prompt_text_len,
        prompt_speech_token,
        prompt_speech_token_len,
        embedding,
        sampling=None,
        max_token_text_ratio=20.0,
        min_token_text_ratio=2.0,
        uuid="",
    ):
        if sampling is None:
            sampling = self._default_sampling

        text_full = torch.cat([prompt_text, text], dim=1) if prompt_text.shape[1] > 0 else text
        text_encoded = self._encode_text(text_full)

        if embedding.shape[0] != 0:
            emb = F.normalize(embedding, dim=1)
            emb = self.spk_embed_affine_layer(emb).unsqueeze(1)
        else:
            emb = torch.zeros(1, 0, 1024, dtype=text_encoded.dtype)

        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token.shape[1] != 0:
            prompt_speech_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_emb = torch.zeros(1, 0, 1024, dtype=text_encoded.dtype)

        lm_input = torch.cat([sos_emb, emb, text_encoded, task_id_emb, prompt_speech_emb], dim=1)

        target_text_len = text_len - (prompt_text_len if prompt_text.shape[1] > 0 else 0)
        min_len = int(target_text_len * min_token_text_ratio)
        max_len = max(min_len + 1, int(target_text_len * max_token_text_ratio))

        out_tokens = []
        offset = 0
        att_cache = None
        x_tt = ttnn.from_torch(lm_input, layout=ttnn.TILE_LAYOUT, device=self.device)
        for i in range(max_len):
            chunk_size = x_tt.shape[1]
            att_mask = torch.tril(torch.ones((1, chunk_size, chunk_size), dtype=torch.bool))
            y_pred, att_cache = self.llm_encoder.forward_chunk(
                x_tt, offset=offset, att_cache=att_cache, att_mask=att_mask
            )
            y_last = ttnn.to_torch(y_pred[:, -1]).float()
            logp = self.llm_decoder(y_last).log_softmax(dim=-1)
            top_id = self._sampling_ids(logp.squeeze(0), out_tokens, sampling, ignore_eos=(i < min_len))
            if top_id == self.eos_token:
                break
            yield top_id
            out_tokens.append(top_id)
            offset += chunk_size
            lm_input = self.speech_embedding.weight[top_id].reshape(1, 1, -1)
            x_tt = ttnn.from_torch(lm_input, layout=ttnn.TILE_LAYOUT, device=self.device)

    def forward(self, batch):
        text_token = batch["text_token"]
        text_token_len = batch["text_token_len"]
        speech_token = batch["speech_token"]
        speech_token_len = batch["speech_token_len"]
        embedding = batch["embedding"]

        text = self.text_embedding(text_token)

        text_tt = ttnn.from_torch(text, layout=ttnn.TILE_LAYOUT, device=self.device)
        mask = torch.ones(1, 1, text.shape[1], dtype=torch.bool)
        text_encoded, _ = self.text_encoder(text_tt, mask)
        text_encoded = ttnn.to_torch(text_encoded).float()

        text = self.text_encoder_affine_layer(text_encoded)

        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        sos_emb = self.llm_embedding.weight[0].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[1].reshape(1, 1, -1)

        speech_emb = self.speech_embedding(speech_token)

        text = unpad_sequence(text, text_token_len.cpu(), batch_first=True)
        speech_emb = unpad_sequence(speech_emb, speech_token_len.cpu(), batch_first=True)

        lm_input = [
            torch.concat(
                [sos_emb.squeeze(dim=0), embedding[i], text[i], task_id_emb.squeeze(dim=0), speech_emb[i]], dim=0
            )
            for i in range(len(text))
        ]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)

        lm_input_tt = ttnn.from_torch(lm_input, layout=ttnn.TILE_LAYOUT, device=self.device)
        mask = torch.ones(1, 1, lm_input.shape[1], dtype=torch.bool)
        lm_output, _ = self.llm_encoder(lm_input_tt, mask)
        lm_output = ttnn.to_torch(lm_output).float()

        logits = self.llm_decoder(lm_output)

        return {"logits": logits}
