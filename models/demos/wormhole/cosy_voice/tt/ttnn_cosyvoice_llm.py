# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the CosyVoice LLM wrapper.

This wraps the Qwen2 backbone with CosyVoice-specific components:
- Speech token embedding
- LLM embedding (SOS/task_id tokens)
- LLM decoder head (logits projection)
- Autoregressive decode loop with top-k sampling
"""

from typing import Dict, Generator, List

import torch

import ttnn
from models.demos.wormhole.cosy_voice.reference.args import CosyVoiceLLMConfig
from models.demos.wormhole.cosy_voice.tt.preprocessing import load_cosyvoice_llm_weights
from models.demos.wormhole.cosy_voice.tt.ttnn_qwen2_model import TtQwen2Model


class TtCosyVoiceLLM(torch.nn.Module):
    """TTNN implementation of CosyVoice's Qwen2-based LLM for speech token generation.

    Pipeline:
        1. Embed input text (via Qwen2 embed_tokens, on host then transfer)
        2. Prepend SOS token, append task_id token
        3. Autoregressive decode: forward through Qwen2 -> project to logits -> sample
        4. Yield speech tokens one by one
    """

    def __init__(
        self,
        device: ttnn.Device,
        configs: dict,
        llm_config: CosyVoiceLLMConfig,
        llm_state_dict: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.device = device
        self.configs = configs
        self.llm_config = llm_config

        self.speech_token_size = llm_config.speech_token_size
        self.sos = llm_config.sos_token
        self.task_id = llm_config.task_id_token
        self.eos_token = llm_config.eos_token
        self.fill_token = llm_config.fill_token
        self.stop_token_ids = [self.speech_token_size + i for i in range(3)]

        # Build the Qwen2 backbone
        self.qwen2 = TtQwen2Model(device, configs, llm_state_dict)

        # Load CosyVoice-specific weights
        wrapper_weights = load_cosyvoice_llm_weights(llm_state_dict, device, dtype=configs["dtype"]["weights"])

        self.speech_embedding_weight = wrapper_weights.get("speech_embedding.weight")
        self.llm_embedding_weight = wrapper_weights.get("llm_embedding.weight")
        self.llm_decoder_weight = wrapper_weights.get("llm_decoder.weight")
        self.llm_decoder_bias = wrapper_weights.get("llm_decoder.bias")

        # Keep the original PyTorch embed_tokens for text embedding (host-side)
        # This is lightweight and avoids transferring the large vocab embedding table
        embed_key = "llm.model.model.embed_tokens.weight"
        if embed_key in llm_state_dict:
            self.text_embed_tokens = torch.nn.Embedding.from_pretrained(llm_state_dict[embed_key], freeze=True)
        else:
            self.text_embed_tokens = None

        # Keep PyTorch speech embedding for token lookups during decode (small tensor)
        speech_emb_key = "speech_embedding.weight"
        if speech_emb_key in llm_state_dict:
            self.speech_embedding_torch = llm_state_dict[speech_emb_key].clone()
        else:
            self.speech_embedding_torch = None

        # Keep LLM embedding for SOS/task_id (2 tokens only)
        llm_emb_key = "llm_embedding.weight"
        if llm_emb_key in llm_state_dict:
            self.llm_embedding_torch = llm_state_dict[llm_emb_key].clone()
        else:
            self.llm_embedding_torch = None

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[int, None, None]:
        """Autoregressive speech token generation.

        Args:
            text: Input text token IDs (1, text_len)
            text_len: Length of input text
            prompt_text: Prompt text token IDs (1, prompt_text_len)
            prompt_text_len: Length of prompt text
            prompt_speech_token: Prompt speech tokens for zero-shot (1, N)
            prompt_speech_token_len: Length of prompt speech tokens
            embedding: Speaker embedding (1, 192) — unused in Qwen2LM variant
            sampling: Top-k sampling parameter
            max_token_text_ratio: Max ratio of speech tokens to text tokens
            min_token_text_ratio: Min ratio for EOS suppression

        Yields:
            Speech token IDs (integers)
        """
        # 1. Prepare text embeddings on host
        text_combined = torch.concat([prompt_text, text], dim=1)
        text_len_combined = text_len + prompt_text_len
        text_emb = self.text_embed_tokens(text_combined)  # (1, total_text_len, 896)

        # 2. Prepare special embeddings
        sos_emb = self.llm_embedding_torch[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding_torch[self.task_id].reshape(1, 1, -1)

        # 3. Prepare prompt speech token embeddings
        if prompt_speech_token_len.item() != 0:
            prompt_speech_emb = self.speech_embedding_torch[prompt_speech_token.flatten()].unsqueeze(0)
        else:
            prompt_speech_emb = torch.zeros(1, 0, self.llm_config.llm_input_size)

        # 4. Concatenate initial LM input: [SOS, text_emb, task_id, prompt_speech_emb]
        lm_input_torch = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_emb], dim=1)

        # 5. Calculate min/max decode length
        min_len = int((text_len_combined - prompt_text_len).item() * min_token_text_ratio)
        max_len = int((text_len_combined - prompt_text_len).item() * max_token_text_ratio)

        # 6. Transfer initial input to device and run autoregressive decode
        out_tokens: List[int] = []
        kv_caches = None

        for i in range(max_len):
            # Convert current input to TTNN
            lm_input_4d = lm_input_torch.unsqueeze(0)  # (1, 1, S, 896)
            tt_input = ttnn.from_torch(
                lm_input_4d,
                dtype=self.configs["dtype"]["activations"],
                layout=ttnn.TILE_LAYOUT,
            )
            tt_input = ttnn.to_device(tt_input, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Forward through Qwen2
            tt_output, kv_caches = self.qwen2.forward_one_step(tt_input, kv_caches=kv_caches)

            # Project to logits (last token only)
            tt_logits = ttnn.linear(
                tt_output[:, :, -1:, :],
                self.llm_decoder_weight,
                bias=self.llm_decoder_bias,
            )

            # Transfer logits to CPU for sampling
            logits_torch = ttnn.to_torch(tt_logits).squeeze()
            logp = torch.log_softmax(logits_torch, dim=-1)

            # Suppress EOS if below minimum length
            if i < min_len:
                for stop_id in self.stop_token_ids:
                    if stop_id < logp.shape[0]:
                        logp[stop_id] = -float("inf")

            # Top-k sampling
            top_ids = self._sample_top_k(logp, k=sampling)

            # Check for stop tokens
            if top_ids in self.stop_token_ids:
                break

            # Yield the generated token
            yield top_ids

            out_tokens.append(top_ids)

            # Prepare next input: embed the generated speech token
            next_emb = self.speech_embedding_torch[top_ids].reshape(1, 1, -1)
            lm_input_torch = next_emb  # Single token for next step

            # Clean up device tensors
            ttnn.deallocate(tt_input)
            ttnn.deallocate(tt_output)
            ttnn.deallocate(tt_logits)

    @staticmethod
    def _sample_top_k(logp: torch.Tensor, k: int = 25) -> int:
        """Top-k sampling from log-probabilities.

        Args:
            logp: Log-probability tensor (vocab_size,)
            k: Number of top candidates to sample from

        Returns:
            Sampled token ID
        """
        top_k_logp, top_k_indices = torch.topk(logp, k)
        probs = torch.softmax(top_k_logp, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        return top_k_indices[sampled_idx].item()
