# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the CosyVoice Flow Matching decoder.

This module converts semantic speech tokens into mel spectrograms using
Conditional Flow Matching (CFM) with an Euler ODE solver.

CosyVoice3 architecture (CausalMaskedDiffWithDiT):
1. Token embedding + speaker projection
2. Pre-lookahead layer (causal conv)
3. Repeat-interleave for token→mel upsampling
4. DiT-based CFM estimator (10 Euler steps, classifier-free guidance)

CosyVoice2 architecture (CausalMaskedDiffWithXvec):
1. Token embedding + speaker projection
2. Conformer encoder
3. Length regulator for token→mel alignment
4. U-Net/ConformerDecoder-based CFM estimator
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

import ttnn


class TtFlowMatching(torch.nn.Module):
    """TTNN implementation of CosyVoice Flow Matching decoder.

    This is a hybrid implementation: the compute-intensive CFM Euler solver
    runs on TTNN, while some preprocessing stays on CPU for simplicity.

    For Stage 1, the DiT estimator forward pass runs on device.
    """

    def __init__(
        self,
        device: ttnn.Device,
        configs: dict,
        flow_state_dict: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.device = device
        self.configs = configs

        self.input_size = configs["input_size"]  # 512
        self.output_size = configs["output_size"]  # 80 (mel channels)
        self.vocab_size = configs["vocab_size"]  # 6561
        self.n_timesteps = configs["n_timesteps"]  # 10
        self.spk_embed_dim = configs.get("spk_embed_dim", 192)
        self.token_mel_ratio = configs.get("token_mel_ratio", 2)
        self.pre_lookahead_len = configs.get("pre_lookahead_len", 3)
        self.inference_cfg_rate = 0.7  # Classifier-free guidance rate

        # Load weights on CPU for initial bring-up
        # The heavy compute (estimator forward) will migrate to TTNN iteratively
        self._load_weights(flow_state_dict)

    def _load_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load flow matching weights. Keep on CPU for Stage 1 initial testing."""
        # Token embedding
        emb_key = "input_embedding.weight"
        if emb_key in state_dict:
            self.input_embedding = torch.nn.Embedding.from_pretrained(state_dict[emb_key], freeze=True)
        else:
            self.input_embedding = torch.nn.Embedding(self.vocab_size, self.input_size)

        # Speaker embedding projection
        spk_key = "spk_embed_affine_layer.weight"
        spk_bias_key = "spk_embed_affine_layer.bias"
        if spk_key in state_dict:
            self.spk_embed_affine_layer = torch.nn.Linear(self.spk_embed_dim, self.output_size)
            self.spk_embed_affine_layer.weight = torch.nn.Parameter(state_dict[spk_key])
            if spk_bias_key in state_dict:
                self.spk_embed_affine_layer.bias = torch.nn.Parameter(state_dict[spk_bias_key])
        else:
            self.spk_embed_affine_layer = torch.nn.Linear(self.spk_embed_dim, self.output_size)

        # Store full state dict for the estimator (will be loaded to device later)
        self._estimator_state_dict = {k: v for k, v in state_dict.items() if "decoder" in k or "estimator" in k}
        self._encoder_state_dict = {
            k: v for k, v in state_dict.items() if "encoder" in k or "pre_lookahead" in k or "encoder_proj" in k
        }

    @torch.inference_mode()
    def inference(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        streaming: bool = False,
        finalize: bool = True,
    ) -> Tuple[torch.Tensor, None]:
        """Convert speech tokens to mel spectrogram via flow matching.

        For Stage 1 bring-up, this runs the CFM solver on CPU with the
        PyTorch reference implementation. The estimator forward pass will
        be migrated to TTNN iteratively.

        Args:
            token: Speech token IDs (1, token_len)
            token_len: Token sequence length
            prompt_token: Prompt speech tokens (1, prompt_token_len)
            prompt_token_len: Prompt token length
            prompt_feat: Prompt mel features (1, prompt_feat_len, 80)
            prompt_feat_len: Prompt feature length
            embedding: Speaker embedding (1, 192)
            streaming: Whether to use streaming mode
            finalize: Whether this is the final chunk

        Returns:
            Tuple of (mel_spectrogram, cache) where mel is (1, 80, mel_len)
        """
        # 1. Speaker embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # 2. Token embedding
        token_combined = torch.concat([prompt_token, token], dim=1)
        token_len_combined = prompt_token_len + token_len
        mask = torch.ones(1, token_combined.shape[1], 1)
        token_emb = self.input_embedding(torch.clamp(token_combined, min=0)) * mask

        # 3. Encoder / pre-lookahead processing
        # For Stage 1 bring-up, use simple repeat-interleave as upsampling
        # The full conformer/pre-lookahead encoder will be added in Phase 2 optimization
        h = token_emb  # (1, total_tokens, input_size)

        # Simple projection to output_size if needed
        if h.shape[-1] != self.output_size:
            # Placeholder: linear projection (will be replaced with proper encoder)
            h = h[:, :, : self.output_size]
            if h.shape[-1] < self.output_size:
                h = F.pad(h, (0, self.output_size - h.shape[-1]))

        # 4. Upsample tokens to mel frame rate
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1

        # 5. Prepare conditions
        conds = torch.zeros(1, mel_len1 + mel_len2, self.output_size)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)  # (1, 80, mel_len)

        # 6. Run CFM Euler solver (CPU for Stage 1)
        mu = h.transpose(1, 2).contiguous()  # (1, 80, mel_len)
        mask_t = torch.ones(1, 1, mel_len1 + mel_len2)

        feat = self._euler_solve(
            mu=mu,
            mask=mask_t,
            spks=embedding,
            cond=conds,
            n_timesteps=self.n_timesteps,
        )

        # 7. Extract generated portion (exclude prompt)
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2

        return feat.float(), None

    def _euler_solve(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """Fixed Euler ODE solver for the flow matching process.

        This is the hot loop that will be migrated to TTNN in optimization phase.

        Args:
            mu: Encoder output (1, 80, mel_len)
            mask: Output mask (1, 1, mel_len)
            spks: Speaker embedding (1, 80)
            cond: Conditioning signal (1, 80, mel_len)
            n_timesteps: Number of Euler solver steps

        Returns:
            Generated mel spectrogram (1, 80, mel_len)
        """
        z = torch.randn_like(mu)

        # Cosine time schedule
        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        x = z
        t, dt = t_span[0], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            # Classifier-free guidance: run estimator with batch=2
            # [guided, unguided] - guided uses real conditions, unguided uses zeros
            x_in = x.repeat(2, 1, 1)
            mask_in = mask.repeat(2, 1, 1)
            mu_in = torch.zeros_like(x_in)
            mu_in[0] = mu
            t_in = t.unsqueeze(0).repeat(2)
            spks_in = torch.zeros(2, spks.shape[1])
            spks_in[0] = spks
            cond_in = torch.zeros_like(x_in)
            cond_in[0] = cond

            # Estimator forward (placeholder — returns simple interpolation for now)
            # TODO: Replace with actual DiT estimator on TTNN
            dphi_dt = self._estimator_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            # Split guided/unguided and apply CFG
            dphi_guided, dphi_unguided = dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_guided - self.inference_cfg_rate * dphi_unguided

            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x.float()

    def _estimator_forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Placeholder for the DiT estimator forward pass.

        In the full implementation, this will run the DiT transformer on TTNN.
        For Stage 1 bring-up, returns a simple velocity estimate:
            v = mu - x (points from noise toward target)

        This ensures the Euler solver converges to something reasonable
        for pipeline integration testing.

        Args:
            x: Noisy input (2, 80, mel_len)
            mask: Mask (2, 1, mel_len)
            mu: Target/encoder output (2, 80, mel_len)
            t: Timestep (2,)
            spks: Speaker embeddings (2, 80)
            cond: Conditioning (2, 80, mel_len)

        Returns:
            Estimated velocity field (2, 80, mel_len)
        """
        # Simple linear velocity: points from noise (x) toward target (mu)
        # This is the optimal transport ODE solution
        return mu - x
