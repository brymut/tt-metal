# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

# Ensure reference paths for lazy imports
cosyvoice_ref = str(Path(__file__).parent.parent / "reference" / "CosyVoice")
matcha_ref = str(Path(__file__).parent.parent / "reference" / "CosyVoice" / "third_party" / "Matcha-TTS")
if cosyvoice_ref not in sys.path:
    sys.path.insert(0, cosyvoice_ref)
if matcha_ref not in sys.path:
    sys.path.insert(0, matcha_ref)

from models.demos.wormhole.cosyvoice.tt.cosyvoice_llm import TtTransformerEncoder

# ------------------------------- TtFlowEncoder -------------------------------


class TtFlowEncoder(nn.Module):
    def __init__(self, device, state_dict):
        super().__init__()
        config = {
            "num_blocks": 6,
            "attention_heads": 8,
            "output_size": 512,
            "input_size": 512,
            "linear_units": 2048,
            "dropout_rate": 0.1,
            "activation_type": "swish",
            "use_relu": False,
            "encoder_prefix": "encoder",
            "norm1_key": "norm_mha",
            "norm2_key": "norm_ff",
        }
        self.encoder = TtTransformerEncoder(device, state_dict, config)

    def forward(self, x, token_len=None):
        return self.encoder(x, mask=None)


# ------------------------------- TtInterpolateRegulator -------------------------------


class TtInterpolateRegulator(nn.Module):
    def __init__(self, device, state_dict=None, ref_regulator=None):
        super().__init__()
        self.device = device
        self._state_dict = state_dict
        self._cpu_model = ref_regulator

    def _ensure_cpu_model(self):
        if self._cpu_model is not None:
            return
        # Fallback: rebuild from state dict.
        from cosyvoice.flow.length_regulator import InterpolateRegulator

        self._cpu_model = InterpolateRegulator(channels=80, sampling_ratios=(1, 1, 1, 1), out_channels=80)
        sd = {
            k.replace("length_regulator.", ""): v
            for k, v in self._state_dict.items()
            if k.startswith("length_regulator.")
        }
        self._cpu_model.load_state_dict(sd, strict=False)
        self._cpu_model.eval()

    def _to_torch(self, x):
        if x is None:
            return None
        if isinstance(x, ttnn.Tensor):
            return ttnn.to_torch(x).float()
        return x.float()

    def forward(self, x, ylens=None):
        self._ensure_cpu_model()
        x_torch = self._to_torch(x)
        out, olens = self._cpu_model(x_torch, ylens)
        out_tt = ttnn.from_torch(out, layout=ttnn.TILE_LAYOUT, device=self.device)
        return out_tt, olens

    def inference(self, x1, x2, mel_len1, mel_len2, input_frame_rate=50):
        self._ensure_cpu_model()
        x1_torch = self._to_torch(x1)
        x2_torch = self._to_torch(x2)
        out, total_len = self._cpu_model.inference(x1_torch, x2_torch, mel_len1, mel_len2, input_frame_rate)
        out_tt = ttnn.from_torch(out, layout=ttnn.TILE_LAYOUT, device=self.device)
        return out_tt, total_len


# ------------------------------- TtConditionalCFM (native on-device) -------------------------------


# Helper functions for input layout conversion
def _ensure_4d_mask(mask: ttnn.Tensor) -> ttnn.Tensor:
    """`mask` may be [B, 1, T] (3D) or [B, 1, T, 1] (4D). Return 4D."""
    if mask.shape[-1] != 1:
        # Add trailing singleton dim
        return ttnn.reshape(mask, (mask.shape[0], mask.shape[1], mask.shape[2], 1))
    return mask


def _ensure_spks_4d(spks: ttnn.Tensor, T: int) -> ttnn.Tensor:
    """`spks` may be [B, 80] (2D) or [B, 1, 80] (3D). Return [B, 1, T, 80]."""
    spks_t = ttnn.to_torch(spks).float()  # [B, 80] or [B, 1, 80]
    if spks_t.ndim == 2:
        spks_t = spks_t.unsqueeze(1)  # [B, 1, 80]
    B = spks_t.shape[0]
    spks_t = spks_t.unsqueeze(2).expand(B, 1, T, spks_t.shape[-1]).contiguous()
    return ttnn.from_torch(spks_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=spks.device())


class TtConditionalCFM(nn.Module):
    """TTNN native Conditional Flow Matching solver.

    Replaces the previous CPU-fallback wrapper. The Euler loop runs in Python
    but each `forward_estimator` call executes the UNet end-to-end on device.
    Classifier-Free Guidance (CFG) is implemented by running the UNet on a
    2x batch ([conditioned, unconditioned]) and combining the predictions.
    """

    def __init__(self, device, state_dict=None, ref_cfm=None):
        super().__init__()
        self.device = device
        # Lazy import to avoid a hard dependency at module import time
        from models.demos.wormhole.cosyvoice.tt.cosyvoice_unet import TtConditionalDecoder

        # The UNet state dict lives under `decoder.estimator.*` in the flow
        # state dict. The reference CFM exposes the estimator as
        # `self.estimator` (we don't depend on that here — pass state_dict
        # explicitly via TtCosyVoiceFlow).
        if state_dict is not None:
            prefix = "decoder.estimator."
            estimator_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            self.estimator = TtConditionalDecoder(device, state_dict=estimator_sd, base_address="")
        else:
            self.estimator = TtConditionalDecoder(device, state_dict=ref_cfm.estimator.state_dict(), base_address="")
        # CFG / solver parameters (read from reference CFM if available).
        if ref_cfm is not None:
            self.sigma_min = ref_cfm.sigma_min
            self.inference_cfg_rate = ref_cfm.inference_cfg_rate
        else:
            self.sigma_min = 1e-6
            self.inference_cfg_rate = 0.7

    @staticmethod
    def _stack_tt(a: ttnn.Tensor, b: ttnn.Tensor, dim: int = 0) -> ttnn.Tensor:
        """Stack two ttnn tensors along a given dim (default 0 = batch)."""
        return ttnn.concat([a, b], dim=dim)

    def _estimator_forward(
        self,
        x_tt: ttnn.Tensor,  # [B, 1, T, 80]  (B is the loop batch — 1 for the call)
        mask_4d_tt: ttnn.Tensor,  # [B, 1, T, 1]
        mu_tt: ttnn.Tensor,  # [B, 1, 80, T]
        t_t: torch.Tensor,  # [B]
        spks_4d_tt: ttnn.Tensor,  # [B, 1, T, 80]
        cond_4d_tt: ttnn.Tensor,  # [B, 1, 80, T]
    ) -> ttnn.Tensor:
        """Run the native UNet once on a single batch (no CFG doubling)."""
        return self.estimator.forward(
            x=x_tt,
            mask=mask_4d_tt,
            mu=mu_tt,
            t=t_t,
            spks=spks_4d_tt,
            cond=cond_4d_tt,
        )

    def _run_unet_2x_cfg(
        self,
        x: ttnn.Tensor,  # [1, 1, T, 80] (single batch)
        mu: ttnn.Tensor,  # [1, 1, 80, T]
        mask_4d: ttnn.Tensor,  # [1, 1, T, 1]
        spks_4d: ttnn.Tensor,  # [1, 1, T, 80]
        cond_4d: ttnn.Tensor,  # [1, 1, 80, T]
        t: torch.Tensor,  # [1]
    ) -> ttnn.Tensor:
        """Run UNet on a 2x batch [conditioned, unconditioned] and apply CFG.

        Returns [1, 1, T, 80] (single batch, CFG-combined).

        When `inference_cfg_rate == 0`, skip the 2x batch and just run the
        conditioned forward once (the uncond half would be wasted compute, and
        the math is identical: `(1 + 0) * dphi - 0 * dphi_uncond = dphi`).
        """
        if self.inference_cfg_rate == 0.0:
            return self.estimator.forward(
                x=x,
                mask=mask_4d,
                mu=mu,
                t=t,
                spks=spks_4d,
                cond=cond_4d,
            )
        # Build the 2x batch. Per the reference solve_euler
        # (flow_matching.py:95-108): x, mask, and t are the SAME in both halves,
        # while mu/spks/cond are non-zero only in the conditioned half (batch 0)
        # and zero in the unconditioned half (batch 1). Previously `mu` was
        # stacked identically into both halves, which made the "unconditioned"
        # half not actually unconditioned and broke the CFG combination.
        x_2x = self._stack_tt(x, x)  # [2, 1, T, 80] (same x in both halves)
        mask_2x = self._stack_tt(mask_4d, mask_4d)  # [2, 1, T, 1]
        mu_zero = ttnn.zeros_like(mu)
        mu_2x = self._stack_tt(mu, mu_zero)  # [2, 1, 80, T] (uncond half zeroed)
        # Unconditioned half: spks and cond set to zero (matches reference where
        # the uncond row of spks_in/cond_in is left at its initial zero value)
        spks_zero = ttnn.zeros_like(spks_4d)
        cond_zero = ttnn.zeros_like(cond_4d)
        spks_2x = self._stack_tt(spks_4d, spks_zero)
        cond_2x = self._stack_tt(cond_4d, cond_zero)
        t_2x = torch.cat([t, t], dim=0)  # [2] (same t in both halves)

        out_2x = self.estimator.forward(
            x=x_2x,
            mask=mask_2x,
            mu=mu_2x,
            t=t_2x,
            spks=spks_2x,
            cond=cond_2x,
        )  # [2, 1, T, 80]
        # Split and apply CFG. ttnn.slice gives us the two halves.
        B2, _, T_dim, C = out_2x.shape
        dphi = ttnn.slice(out_2x, [0, 0, 0, 0], [1, 1, T_dim, C])
        dphi_uncond = ttnn.slice(out_2x, [1, 0, 0, 0], [2, 1, T_dim, C])
        # CFG: (1 + r) * cond - r * uncond
        dphi = ttnn.multiply(dphi, 1.0 + self.inference_cfg_rate)
        dphi_uncond = ttnn.multiply(dphi_uncond, self.inference_cfg_rate)
        dphi = ttnn.subtract(dphi, dphi_uncond)
        return dphi

    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=None):
        """Solve the flow ODE with on-device UNet + CFG.

        All inputs are ttnn tensors in the layouts produced by
        `TtCosyVoiceFlow.inference`. The 10-step Euler loop runs in Python;
        each UNet call runs end-to-end on device.
        """
        T = mu.shape[3]
        device = self.device

        # z = torch.randn_like(mu) — CPU then to device (single round-trip)
        mu_t = ttnn.to_torch(mu).float()  # [B, 1, 80, T]
        z_t = torch.randn_like(mu_t) * temperature
        z_tt = ttnn.from_torch(z_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # UNet expects x in [B, 1, T, C] layout (T is dim 2, C is last).
        # Permute the freshly-created z from [B, 1, 80, T] to [B, 1, T, 80].
        z_tt = ttnn.permute(z_tt, (0, 1, 3, 2))
        # Build the time span (cosine scheduler, CPU)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device="cpu", dtype=torch.float32)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Convert mask / spks / cond to the [B, 1, T, ...] layouts the UNet expects
        mask_4d = _ensure_4d_mask(mask)  # [B, 1, T, 1]
        spks_4d = _ensure_spks_4d(spks, T)  # [B, 1, T, 80]
        cond_4d = cond  # already [B, 1, 80, T]

        # Euler loop on device. The running state `x` is kept in fp32 to avoid
        # systematic bf16 rounding of the accumulator itself: at step 10 |x|~6,
        # so each bf16 rounding of x is ~0.023, a non-trivial drift over 10
        # steps that manifests as the spectral-STD collapse (Session 12). The
        # UNet still receives bf16 input (its weights/activations are bf16);
        # only the accumulation x_{n+1} = x_n + dt*dphi_n is done in fp32.
        # dphi returns bf16 from the UNet and is upcast for the add.
        x = ttnn.typecast(z_tt, dtype=ttnn.float32)
        for step in range(1, n_timesteps + 1):
            t_cur = t_span[step - 1].unsqueeze(0)  # [1]
            x_bf16 = ttnn.typecast(x, dtype=ttnn.bfloat16)
            dphi = self._run_unet_2x_cfg(
                x=x_bf16,
                mu=mu,
                mask_4d=mask_4d,
                spks_4d=spks_4d,
                cond_4d=cond_4d,
                t=t_cur,
            )
            # x = x + dt * dphi  (fp32 accumulation)
            dt = (t_span[step] - t_span[step - 1]).item()
            dphi_f = ttnn.typecast(dphi, dtype=ttnn.float32)
            dphi_scaled = ttnn.multiply(dphi_f, dt)
            x = ttnn.add(x, dphi_scaled)
            ttnn.deallocate(dphi)
            ttnn.deallocate(dphi_f)
            ttnn.deallocate(dphi_scaled)
            ttnn.deallocate(x_bf16)

        out_tt = x
        # Cache: return a zero cache (we don't use streaming cache here)
        cache = torch.zeros(1, 80, 0, 2, dtype=torch.float32)
        return out_tt, cache


# ------------------------------- Top-Level TtCosyVoiceFlow -------------------------------


class TtCosyVoiceFlow(nn.Module):
    def __init__(self, device, state_dict=None, ref_flow=None):
        super().__init__()
        self.device = device
        self.state_dict = state_dict

        from cosyvoice.utils.mask import make_pad_mask

        self._make_pad_mask = make_pad_mask

        self.input_embedding = nn.Embedding(4096, 512)
        self.spk_embed_affine_layer = nn.Linear(192, 80)
        self.encoder_proj = nn.Linear(512, 80)

        if state_dict is not None:
            self.input_embedding.weight.data = state_dict["input_embedding.weight"]
            self.spk_embed_affine_layer.weight.data = state_dict["spk_embed_affine_layer.weight"]
            self.spk_embed_affine_layer.bias.data = state_dict["spk_embed_affine_layer.bias"]
            self.encoder_proj.weight.data = state_dict["encoder_proj.weight"]
            self.encoder_proj.bias.data = state_dict["encoder_proj.bias"]

        self.encoder = TtFlowEncoder(device, state_dict)

        ref_reg = ref_flow.length_regulator if ref_flow is not None else None
        self.length_regulator = TtInterpolateRegulator(device, state_dict=state_dict, ref_regulator=ref_reg)

        ref_cfm = ref_flow.decoder if ref_flow is not None else None
        self.decoder = TtConditionalCFM(device, state_dict=state_dict, ref_cfm=ref_cfm)
        # Disable Classifier-Free Guidance for the bring-up stage. CFG doubles
        # the batch through the UNet, which currently triggers a tile-padding
        # broadcasting bug for B=2 (the 2x batch path is the next item to
        # debug — see HANDOFF.md). With CFG off, the Euler loop calls the
        # native UNet once per step on a single batch.
        self.decoder.inference_cfg_rate = 0.0

    def forward(self, batch):
        raise NotImplementedError("Flow forward (training) not yet implemented.")

    def inference(
        self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, flow_cache
    ):
        # CPU-side preprocessing (same as reference MaskedDiffWithXvec.inference)
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token_all = torch.cat([prompt_token, token], dim=1)
        token_len_all = prompt_token_len + token_len
        mask = (~self._make_pad_mask(token_len_all)).unsqueeze(-1).to(embedding)
        token_emb = self.input_embedding(torch.clamp(token_all, min=0)) * mask

        # Send to device for encoder
        token_tt = ttnn.from_torch(token_emb, layout=ttnn.TILE_LAYOUT, device=self.device)
        h, _ = self.encoder(token_tt, token_len=token_len_all)

        # Back to CPU for remaining ops
        h_cpu = ttnn.to_torch(h).float()
        h_cpu = self.encoder_proj(h_cpu)

        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / 50 * 22050 / 256)
        h_reg, _ = self.length_regulator.inference(h_cpu[:, :token_len1], h_cpu[:, token_len1:], mel_len1, mel_len2, 50)

        # Convert back to torch for CPU-side downstream ops
        h_reg_torch = ttnn.to_torch(h_reg).float() if isinstance(h_reg, ttnn.Tensor) else h_reg.float()

        # Build conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, 80], device=embedding.device, dtype=torch.float32)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask_cpu = (~self._make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).float()
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Call decoder (TT native CFM solver)
        feat_tt, flow_cache = self.decoder(
            mu=ttnn.from_torch(
                h_reg_torch.transpose(1, 2).unsqueeze(1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            ),  # [B, 1, 80, T]
            mask=ttnn.from_torch(
                mask_cpu.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),  # [B, 1, T]
            n_timesteps=10,
            spks=ttnn.from_torch(
                embedding.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),  # [B, 1, 80]
            cond=ttnn.from_torch(
                conds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),  # [B, 1, 80, T]
            prompt_len=mel_len1,
            cache=flow_cache,
        )

        feat_cpu = ttnn.to_torch(feat_tt).float()
        feat_cpu = feat_cpu[:, :, mel_len1:]
        assert feat_cpu.shape[2] == mel_len2
        return feat_cpu, flow_cache
