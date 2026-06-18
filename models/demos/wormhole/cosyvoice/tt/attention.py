# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""TT-native multi-head attention layers for the CosyVoice LLM.

The `forward` of `TtRelPositionMultiHeadedAttention` runs the entire attention
math (Q/K/V projections, rel_pos projection, bias add, two matmuls, rel_shift,
softmax, output matmul) on the TT device. Only the per-step cache is
downloaded to host at the end of each call. This is the single biggest lever
for the LLM's per-step bf16 error accumulation (HANDOFF §6 Priority 4):
removing the q/k/v round-trip drops the per-step noise floor and may
eliminate the greedy-token divergence after step 1.
"""

import math

import torch

import ttnn

# HiFi4 = higher precision math. Helps reduce systematic bf16 error in the
# LLM's per-step divergence, and prevents regression when moving attention to device.
# See HANDOFF §3 Session 2 and Session 8.
_ATTN_LINEAR_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
)


class TtRelPositionMultiHeadedAttention(torch.nn.Module):
    def __init__(self, device, state_dict, base_address, n_head, n_feat, dropout_rate=0.0, key_bias=True):
        super().__init__()
        self.device = device
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat

        self.linear_q_weight = state_dict[f"{base_address}.linear_q.weight"]
        self.linear_q_bias = state_dict[f"{base_address}.linear_q.bias"]
        self.linear_k_weight = state_dict[f"{base_address}.linear_k.weight"]
        self.linear_k_bias = state_dict.get(f"{base_address}.linear_k.bias") if key_bias else None
        self.linear_v_weight = state_dict[f"{base_address}.linear_v.weight"]
        self.linear_v_bias = state_dict[f"{base_address}.linear_v.bias"]
        self.linear_out_weight = state_dict[f"{base_address}.linear_out.weight"]
        self.linear_out_bias = state_dict[f"{base_address}.linear_out.bias"]

        self.linear_pos_weight = state_dict[f"{base_address}.linear_pos.weight"]
        self.pos_bias_u = state_dict[f"{base_address}.pos_bias_u"]
        self.pos_bias_v = state_dict[f"{base_address}.pos_bias_v"]

        # Pre-upload weights (transposed for ttnn.linear)
        self.tt_linear_q_weight = ttnn.from_torch(
            self.linear_q_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_q_bias = ttnn.from_torch(
            self.linear_q_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_k_weight = ttnn.from_torch(
            self.linear_k_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        if self.linear_k_bias is not None:
            self.tt_linear_k_bias = ttnn.from_torch(
                self.linear_k_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
        else:
            self.tt_linear_k_bias = None
        self.tt_linear_v_weight = ttnn.from_torch(
            self.linear_v_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_v_bias = ttnn.from_torch(
            self.linear_v_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_out_weight = ttnn.from_torch(
            self.linear_out_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_out_bias = ttnn.from_torch(
            self.linear_out_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.tt_linear_pos_weight = ttnn.from_torch(
            self.linear_pos_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # Pre-upload pos_bias_u, pos_bias_v reshaped to (1, h, 1, d_k) so they
        # broadcast against q (B, h, T_q, d_k) after the permute below. The
        # reference adds the bias in (B, T_q, h, d_k) layout, then transposes;
        # we add in (B, h, T_q, d_k) layout directly to save the transpose.
        self.tt_pos_bias_u = ttnn.from_torch(
            self.pos_bias_u.reshape(1, self.h, 1, self.d_k),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.tt_pos_bias_v = ttnn.from_torch(
            self.pos_bias_v.reshape(1, self.h, 1, self.d_k),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Pre-upload scale factor 1/sqrt(d_k) and -inf for masking
        self.tt_scale = ttnn.from_torch(
            torch.tensor(1.0 / math.sqrt(self.d_k), dtype=torch.float32).reshape(1, 1, 1, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.tt_neg_inf = ttnn.from_torch(
            torch.tensor(-float("inf"), dtype=torch.float32).reshape(1, 1, 1, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def _rel_shift_ttnn(self, x):
        """On-device rel_shift mirroring the reference
        ``RelPositionMultiHeadedAttention.rel_shift`` (espnet view-trick):

            zero_pad = zeros((B, h, T_q, 1))
            x_padded = cat([zero_pad, x], dim=-1)        # (B, h, T_q, P+1)
            x_padded = x_padded.view(B, h, P+1, T_q)     # memory reinterpret
            x = x_padded[:, :, 1:].view_as(x)            # (B, h, T_q, P)
            x = x[:, :, :, : P // 2 + 1]                 # (B, h, T_q, P//2+1)

        where ``P = x.size(-1) = 2 * total_len - 1`` and the output keeps
        ``P // 2 + 1 = total_len`` columns (matching ``matrix_ac``'s key axis).

        The previous TT implementation used ``permute`` (a true transpose) and
        sliced ``[:T_q]`` instead of ``[: P//2 + 1]``. For the streaming
        autoregressive path ``T_q == 1`` this collapsed ``matrix_bd`` to a
        single column instead of ``total_len`` columns, which is the root cause
        of the LLM diverging from the reference after the first decoded token.

        ttnn cannot do the ``view`` memory-reinterpretation, so:
          * For ``T_q == 1`` (the only case hit on the on-device streaming
            path) the permute chain degenerates to taking the first
            ``P // 2 + 1`` columns of ``x`` directly -- computed on device.
          * For ``T_q > 1`` (non-streaming; normally routed to ``forward_cpu``)
            we fall back to a host round-trip using the exact reference torch
            ops, which is bit-exact and cheap (small tensor).
        """
        B, H, T_q, P = x.shape
        keep = P // 2 + 1
        if T_q == 1:
            # Streaming single-token chunk: rel_shift reduces to the first
            # `keep` columns (verified bit-exact vs the reference for T_q=1).
            return x[:, :, :, :keep]

        # General case: host round-trip with the reference's exact view-trick.
        xt = ttnn.to_torch(x).float()
        zero_pad = torch.zeros((xt.size(0), xt.size(1), xt.size(2), 1), dtype=xt.dtype)
        x_padded = torch.cat([zero_pad, xt], dim=-1)
        x_padded = x_padded.view(xt.size(0), xt.size(1), xt.size(3) + 1, xt.size(2))
        xt = x_padded[:, :, 1:].view_as(xt)[:, :, :, : xt.size(-1) // 2 + 1]
        return ttnn.from_torch(xt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

    def forward_cpu(self, query, key, value, mask=None, pos_emb=None):
        import torch.nn.functional as F

        query_torch = ttnn.to_torch(query).float()
        key_torch = ttnn.to_torch(key).float()
        value_torch = ttnn.to_torch(value).float()
        pos_emb_torch = pos_emb.float() if not isinstance(pos_emb, ttnn.Tensor) else ttnn.to_torch(pos_emb).float()

        B = query_torch.size(0)

        q = F.linear(query_torch, self.linear_q_weight.float(), self.linear_q_bias.float())
        k = F.linear(
            key_torch,
            self.linear_k_weight.float(),
            self.linear_k_bias.float() if self.linear_k_bias is not None else None,
        )
        v = F.linear(value_torch, self.linear_v_weight.float(), self.linear_v_bias.float())

        q = q.view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, -1, self.h, self.d_k).transpose(1, 2)

        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb_torch.size(0)
        p = F.linear(pos_emb_torch, self.linear_pos_weight.float())
        p = p.view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)

        q_with_bias_u = (q.transpose(1, 2) + self.pos_bias_u.float()).transpose(1, 2)
        q_with_bias_v = (q.transpose(1, 2) + self.pos_bias_v.float()).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        if matrix_ac.shape != matrix_bd.shape:
            zero_pad = torch.zeros(
                (matrix_bd.size(0), matrix_bd.size(1), matrix_bd.size(2), 1),
                device=matrix_bd.device,
                dtype=matrix_bd.dtype,
            )
            x_padded = torch.cat([zero_pad, matrix_bd], dim=-1)
            x_padded = x_padded.view(matrix_bd.size(0), matrix_bd.size(1), matrix_bd.size(3) + 1, matrix_bd.size(2))
            matrix_bd = x_padded[:, :, 1:].view_as(matrix_bd)[:, :, :, : matrix_bd.size(-1) // 2 + 1]

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        if mask is not None and mask.numel() > 0:
            mask_bool = mask.unsqueeze(1).eq(0)
            if mask_bool.size(-1) > scores.size(-1):
                mask_bool = mask_bool[:, :, :, : scores.size(-1)]
            scores = scores.masked_fill(mask_bool, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask_bool, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)

        out = F.linear(x, self.linear_out_weight.float(), self.linear_out_bias.float())
        out_tt = ttnn.from_torch(
            out.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        return out_tt, new_cache

    def forward(self, query, key, value, mask=None, pos_emb=None, cache=None):
        """On-device scaled dot product attention with relative position encoding.

        Args:
            query, key, value: ttnn tensors, shape (B, T, n_feat).
            mask: torch bool tensor (B, T_q, T_kv) or None. True = keep.
                If None or empty, the mask is treated as all-keep.
            pos_emb: torch tensor (B, T_pos, n_feat). Uploaded to device.
            cache: torch tensor (1, h, cache_t, d_k*2) or None. The last dim
                is [k_part, v_part]. Uploaded to device, split, concatenated
                with the new k/v. A new cache is built on device and returned
                as a torch tensor (one host sync per call).

        Returns:
            output: ttnn tensor (B, T_q, n_feat)
            new_cache: torch tensor (1, h, T_kv, d_k*2)
        """
        if cache is None:
            return self.forward_cpu(query, key, value, mask, pos_emb)

        B = query.shape[0]
        T_q = query.shape[1]
        T_kv = key.shape[1]
        n_batch_pos = pos_emb.shape[0]
        T_pos = pos_emb.shape[1]

        # 1. Q/K/V projections (on device, HiFi4). Output fp32 directly so the
        # bf16→fp32 cast at the end of the matmul preserves the full accumulator
        # precision (HiFi4 = fp32 accumulation). With dtype=bf16, the accumulator
        # result is rounded back to bf16, losing 8 mantissa bits per element.
        q = ttnn.linear(
            query,
            self.tt_linear_q_weight,
            bias=self.tt_linear_q_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )
        k = ttnn.linear(
            key,
            self.tt_linear_k_weight,
            bias=self.tt_linear_k_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )
        v = ttnn.linear(
            value,
            self.tt_linear_v_weight,
            bias=self.tt_linear_v_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )

        # 2. Reshape (B, T, n_feat) -> (B, h, T, d_k)
        q = ttnn.reshape(q, (B, T_q, self.h, self.d_k))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (B, T_kv, self.h, self.d_k))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (B, T_kv, self.h, self.d_k))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Cast to fp32 BEFORE the cache concat so the cache is stored in fp32.
        # The cache is downloaded and re-uploaded every step; storing it in bf16
        # loses 8 mantissa bits per element, which compounds over 14 layers and
        # many decode steps and is a significant source of LLM token divergence.
        q = ttnn.typecast(q, dtype=ttnn.float32)
        k = ttnn.typecast(k, dtype=ttnn.float32)
        v = ttnn.typecast(v, dtype=ttnn.float32)

        # 3. Cache concat on device (cache is host torch, upload as fp32 + split + concat)
        if cache is not None and cache.numel() > 0:
            cache_tt = ttnn.from_torch(cache, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            key_cache, value_cache = ttnn.chunk(cache_tt, 2, dim=-1)
            k = ttnn.concat([key_cache, k], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            v = ttnn.concat([value_cache, v], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            # NOTE: don't explicitly deallocate key_cache/value_cache/cache_tt here.
            # ttnn.concat consumes its inputs; explicit deallocates after the fact
            # can cause "Tensor is not allocated" errors in downstream ops.

        # 4. Build new cache: concat k and v along the last dim on device, then
        # one host download. This is the only sync in the attention body.
        new_cache_tt = ttnn.concat([k, v], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        new_cache = ttnn.to_torch(new_cache_tt)  # fp32 (k and v were typecast to fp32 above)
        # NOTE: don't deallocate new_cache_tt; let ttnn's memory management handle it.

        # 5. linear_pos on pos_emb (on device, HiFi4). Upload as fp32 to avoid
        # losing 8 mantissa bits to bf16 in the pos_emb (it is recomputed
        # every step on host, so this costs only the upload bytes).
        pos_emb_tt = ttnn.from_torch(pos_emb, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        p = ttnn.linear(
            pos_emb_tt,
            self.tt_linear_pos_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )
        p = ttnn.reshape(p, (n_batch_pos, T_pos, self.h, self.d_k))
        p = ttnn.permute(p, (0, 2, 1, 3))
        # NOTE: don't deallocate pos_emb_tt explicitly; the reshape/permute chain
        # and ttnn's memory management handle the intermediate buffers.

        # 6. Add pos_bias to q (broadcasts (1, h, 1, d_k) over (B, h, T_q, d_k))
        q_with_bias_u = ttnn.add(q, self.tt_pos_bias_u, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)
        q_with_bias_v = ttnn.add(q, self.tt_pos_bias_v, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)

        # 7. Two matmuls: (B, h, T_q, d_k) @ (B, h, d_k, T_kv|T_pos) -> (B, h, T_q, *)
        matrix_ac = ttnn.matmul(
            q_with_bias_u,
            k,
            transpose_b=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )
        matrix_bd = ttnn.matmul(
            q_with_bias_v,
            p,
            transpose_b=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )
        # NOTE: don't deallocate q_with_bias_u/v explicitly; the matmul consumes them
        # and the add op downstream needs the matmul outputs to be valid.

        # 8. rel_shift if shapes differ (non-streaming path)
        if matrix_ac.shape[-1] != matrix_bd.shape[-1]:
            matrix_bd = self._rel_shift_ttnn(matrix_bd)

        # 9. Add and scale
        scores = ttnn.add(matrix_ac, matrix_bd, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)
        scores = ttnn.multiply(scores, self.tt_scale, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)

        # 10. Apply mask. Two cases:
        #   (a) No mask or all-True mask: skip (zero work).
        #   (b) Mask with some False: ttnn.where(mask, scores, -inf) where
        #       mask is broadcastable from (B, 1, T_q, T_kv) to (B, h, T_q, T_kv).
        if mask is not None and mask.numel() > 0 and mask.shape[1] > 0:
            mask_tt = ttnn.from_torch(
                mask.unsqueeze(1).to(torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            scores = ttnn.where(mask_tt, scores, self.tt_neg_inf, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 11. Softmax + matmul with v
        attn = ttnn.softmax(scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.matmul(
            attn,
            v,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )

        # 12. Reshape (B, h, T_q, d_k) -> (B, T_q, n_feat) for the output linear
        x = ttnn.permute(x, (0, 2, 1, 3))
        x = ttnn.reshape(x, (B, T_q, self.n_feat))
        # Keep x in fp32 — the output linear takes bf16 weight but accepts fp32 input
        # (internally downcasts). Keeping the matmul result in fp32 preserves the
        # precision of the attention output before the output projection.

        # 13. Output projection (on device, HiFi2)
        output = ttnn.linear(
            x,
            self.tt_linear_out_weight,
            bias=self.tt_linear_out_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )

        return output, new_cache


class TtMultiHeadedAttention(torch.nn.Module):
    def __init__(self, device, state_dict, base_address, n_head, n_feat, dropout_rate=0.0, key_bias=True):
        super().__init__()
        self.device = device
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat

        self.linear_q_weight = state_dict[f"{base_address}.linear_q.weight"]
        self.linear_q_bias = state_dict[f"{base_address}.linear_q.bias"]
        self.linear_k_weight = state_dict[f"{base_address}.linear_k.weight"]
        self.linear_k_bias = state_dict.get(f"{base_address}.linear_k.bias") if key_bias else None
        self.linear_v_weight = state_dict[f"{base_address}.linear_v.weight"]
        self.linear_v_bias = state_dict[f"{base_address}.linear_v.bias"]
        self.linear_out_weight = state_dict[f"{base_address}.linear_out.weight"]
        self.linear_out_bias = state_dict[f"{base_address}.linear_out.bias"]

        self.tt_linear_q_weight = ttnn.from_torch(
            self.linear_q_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_q_bias = ttnn.from_torch(
            self.linear_q_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_k_weight = ttnn.from_torch(
            self.linear_k_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        if self.linear_k_bias is not None:
            self.tt_linear_k_bias = ttnn.from_torch(
                self.linear_k_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
        else:
            self.tt_linear_k_bias = None
        self.tt_linear_v_weight = ttnn.from_torch(
            self.linear_v_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_v_bias = ttnn.from_torch(
            self.linear_v_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_out_weight = ttnn.from_torch(
            self.linear_out_weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tt_linear_out_bias = ttnn.from_torch(
            self.linear_out_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward_qkv(self, query, key, value):
        n_batch = query.shape[0]

        q = ttnn.linear(
            query,
            self.tt_linear_q_weight,
            bias=self.tt_linear_q_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )
        k = ttnn.linear(
            key,
            self.tt_linear_k_weight,
            bias=self.tt_linear_k_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )
        v = ttnn.linear(
            value,
            self.tt_linear_v_weight,
            bias=self.tt_linear_v_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )

        q_torch = ttnn.to_torch(q)
        k_torch = ttnn.to_torch(k)
        v_torch = ttnn.to_torch(v)

        q_torch = q_torch.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k_torch = k_torch.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v_torch = v_torch.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        return q_torch, k_torch, v_torch

    def forward_attention(self, value, scores, mask=None):
        n_batch = value.size(0)

        if mask is not None and mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:, :, :, : scores.size(-1)]
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)

        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        output = ttnn.linear(
            x_tt,
            self.tt_linear_out_weight,
            bias=self.tt_linear_out_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_ATTN_LINEAR_KERNEL_CFG,
        )
        ttnn.deallocate(x_tt)

        return output

    def forward(self, query, key, value, mask=None, pos_emb=None, cache=None):
        q, k, v = self.forward_qkv(query, key, value)

        if cache is not None and cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)

        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask), new_cache
