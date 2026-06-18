# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.demos.wormhole.cosyvoice.tt.cosyvoice_flow import TtCosyVoiceFlow
from models.demos.wormhole.cosyvoice.tt.cosyvoice_hifigan import TtCosyVoiceHiFiGAN, deparametrize_weight_norm
from models.demos.wormhole.cosyvoice.tt.cosyvoice_llm import TtCosyVoiceLLM


class TtCosyVoiceModel(torch.nn.Module):
    """End-to-end CosyVoice-300M model on TTNN.

    Chains the three sub-pipelines:
        LLM.inference()    -> speech tokens         (TT native + KV-cache)
        Flow.inference()   -> mel spectrogram        (TT native, with CPU regulator)
        HiFiGAN.inference() -> waveform              (TT native, CPU-fallback path for resblocks)

    Supports all 4 official inference modes (SFT, zero-shot, cross-lingual, instruct).
    Per-mode input dicts are produced by ``reference.golden_pipeline.build_test_case``
    and mirror the official CosyVoice frontend (sft / zero_shot / cross_lingual / instruct).

    Args:
        device: TTNN device.
        config: model config dict (from `create_model_config`).
        args: CosyVoiceArgs (currently unused; kept for signature parity).
        state_dict: Optional flat state dict (used only if `ref_model` is None).
        ref_model: Reference `CosyVoiceReferenceModel` instance. Used to pass
            `ref_model.llm.state_dict()`, `ref_model.flow.state_dict()`,
            and `ref_model.hifigan.state_dict()` to the three sub-modules, and
            to pass `ref_model.hifigan` to the HiFi-GAN CPU-fallback path. The
            device-side `resblocks` path is currently blocked by L1 overflow
            at T~1152 — see HANDOFF §5.
    """

    SAMPLE_RATE = 22050
    MEL_DIM = 80
    INPUT_FRAME_RATE = 50
    SPK_EMBED_DIM = 192

    def __init__(self, device, config, args, state_dict=None, ref_model=None):
        super().__init__()
        self.device = device
        self.config = config
        self.args = args
        self.ref_model = ref_model

        # Sub-state-dicts: the LLM/Flow/HiFTGenerator state_dicts have
        # overlapping key names (both have `spk_embed_affine_layer.weight`).
        # Each TT module is constructed with its own sub-state-dict, not a
        # combined one.
        llm_sd = ref_model.llm.state_dict() if ref_model is not None else (state_dict or {})
        flow_sd = ref_model.flow.state_dict() if ref_model is not None else (state_dict or {})
        hift_sd = (
            deparametrize_weight_norm(ref_model.hifigan.state_dict())
            if ref_model is not None
            else (deparametrize_weight_norm(state_dict) if state_dict is not None else {})
        )
        cpu_hifigan = ref_model.hifigan if ref_model is not None else None

        self.llm = TtCosyVoiceLLM(device, config, args, state_dict=llm_sd)
        self.flow = TtCosyVoiceFlow(
            device, state_dict=flow_sd, ref_flow=ref_model.flow if ref_model is not None else None
        )
        self.hifigan = TtCosyVoiceHiFiGAN(
            device,
            state_dict=hift_sd,
            base_address="",
            cpu_hifigan=cpu_hifigan,
        )

    def _flow_cache(self) -> torch.Tensor:
        return torch.zeros(1, self.MEL_DIM, 0, 2, dtype=torch.float32)

    @staticmethod
    def _empty_token() -> torch.Tensor:
        return torch.zeros(1, 0, dtype=torch.int32)

    @staticmethod
    def _zero_len() -> torch.Tensor:
        return torch.tensor([0], dtype=torch.int32)

    def _empty_prompt_speech_token(self) -> torch.Tensor:
        return self._empty_token()

    def _empty_prompt_speech_token_len(self) -> torch.Tensor:
        return self._zero_len()

    def _empty_prompt_text(self) -> torch.Tensor:
        return self._empty_token()

    def _empty_prompt_text_len(self) -> torch.Tensor:
        return self._zero_len()

    def _empty_prompt_feat(self) -> torch.Tensor:
        return torch.zeros(1, 0, self.MEL_DIM, dtype=torch.float32)

    def _drop_spk_embedding(self) -> torch.Tensor:
        """Shape (0, 192) — the LLM checks `embedding.shape[0] != 0` to skip
        the `spk_embed_affine_layer` (matches reference's `del llm_embedding` in
        `frontend_instruct`)."""
        return torch.zeros(0, self.SPK_EMBED_DIM, dtype=torch.float32)

    def _generate(self, inputs: dict, max_speech_tokens: int) -> torch.Tensor:
        """Shared E2E driver: LLM → Flow → HiFi-GAN.

        The input dict must match the format produced by
        ``reference.golden_pipeline.build_test_case``:
            text, text_len,
            prompt_text, prompt_text_len,
            llm_prompt_speech_token, llm_prompt_speech_token_len,
            flow_prompt_speech_token, flow_prompt_speech_token_len,
            prompt_speech_feat, prompt_speech_feat_len,
            llm_embedding, flow_embedding
        """
        # 1. LLM autoregressive decode -> speech tokens
        speech_tokens: list[int] = []
        for tok in self.llm.inference(
            text=inputs["text"],
            text_len=inputs["text_len"],
            prompt_text=inputs["prompt_text"],
            prompt_text_len=inputs["prompt_text_len"],
            prompt_speech_token=inputs["llm_prompt_speech_token"],
            prompt_speech_token_len=inputs["llm_prompt_speech_token_len"],
            embedding=inputs["llm_embedding"],
            sampling=25,
            max_token_text_ratio=20.0,
            min_token_text_ratio=2.0,
        ):
            speech_tokens.append(int(tok))
            if len(speech_tokens) >= max_speech_tokens:
                break
        if not speech_tokens:
            raise RuntimeError("LLM produced no speech tokens")
        speech_tokens_t = torch.tensor([speech_tokens], dtype=torch.int32)

        # 2. Flow: tokens + prompt mel + spk -> mel spectrogram
        with torch.no_grad():
            mel, _ = self.flow.inference(
                token=speech_tokens_t,
                token_len=torch.tensor([speech_tokens_t.shape[1]], dtype=torch.int32),
                prompt_token=inputs["flow_prompt_speech_token"],
                prompt_token_len=inputs["flow_prompt_speech_token_len"],
                prompt_feat=inputs["prompt_speech_feat"],
                prompt_feat_len=inputs["prompt_speech_feat_len"],
                embedding=inputs["flow_embedding"],
                flow_cache=self._flow_cache(),
            )

        # The TT flow returns mel in the UNet's 4D [B, 1, T, 80] layout; the
        # HiFi-GAN (and reference) expects 3D [B, 80, T].
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        if mel.dim() == 3 and mel.shape[1] != self.MEL_DIM and mel.shape[2] == self.MEL_DIM:
            mel = mel.transpose(1, 2).contiguous()

        # 3. HiFi-GAN: mel -> waveform
        wav = self.hifigan.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))
        return wav

    @torch.inference_mode()
    def inference_sft(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        llm_embedding: torch.Tensor,
        max_speech_tokens: int = 50,
    ) -> torch.Tensor:
        """SFT-mode TTS (predefined speaker): text -> wav.

        Mirrors the official ``frontend_sft``: the LLM does NOT see prompt_text
        or prompt_speech_token; only ``llm_embedding`` (a real speaker embedding
        from ``spk2info.pt``) is used. The flow also reuses the same embedding
        and gets zero prompt mel.

        Args:
            text: [1, T_text] int32 whisper tokens.
            text_len: [1] int32 text length.
            llm_embedding: [1, 192] real speaker embedding.
            max_speech_tokens: cap on the number of LLM-generated speech tokens.
        """
        inputs = {
            "text": text,
            "text_len": text_len,
            "prompt_text": self._empty_prompt_text(),
            "prompt_text_len": self._empty_prompt_text_len(),
            "llm_prompt_speech_token": self._empty_prompt_speech_token(),
            "llm_prompt_speech_token_len": self._empty_prompt_speech_token_len(),
            "flow_prompt_speech_token": self._empty_prompt_speech_token(),
            "flow_prompt_speech_token_len": self._empty_prompt_speech_token_len(),
            "prompt_speech_feat": self._empty_prompt_feat(),
            "prompt_speech_feat_len": self._zero_len(),
            "llm_embedding": llm_embedding,
            "flow_embedding": llm_embedding,
        }
        return self._generate(inputs, max_speech_tokens)

    @torch.inference_mode()
    def inference_zero_shot(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        llm_prompt_speech_token: torch.Tensor,
        llm_prompt_speech_token_len: torch.Tensor,
        flow_prompt_speech_token: torch.Tensor,
        flow_prompt_speech_token_len: torch.Tensor,
        prompt_speech_feat: torch.Tensor,
        prompt_speech_feat_len: torch.Tensor,
        llm_embedding: torch.Tensor,
        flow_embedding: torch.Tensor | None = None,
        max_speech_tokens: int = 50,
    ) -> torch.Tensor:
        """Zero-shot mode (voice cloning): synthesize speech in the voice of a reference
        audio clip. The LLM sees the prompt text + prompt speech tokens; the flow uses
        prompt mel + speaker embedding.

        Mirrors the official ``frontend_zero_shot``. The LLM needs the prompt text
        tokens (so it can align semantics with the prompt audio) and the prompt speech
        tokens. The flow needs the prompt mel and a speaker embedding. By convention
        the prompt mel and llm_embedding are derived from the SAME reference audio
        (so they share an identity); if `flow_embedding` is None we reuse `llm_embedding`.
        """
        if flow_embedding is None:
            flow_embedding = llm_embedding
        inputs = {
            "text": text,
            "text_len": text_len,
            "prompt_text": prompt_text,
            "prompt_text_len": prompt_text_len,
            "llm_prompt_speech_token": llm_prompt_speech_token,
            "llm_prompt_speech_token_len": llm_prompt_speech_token_len,
            "flow_prompt_speech_token": flow_prompt_speech_token,
            "flow_prompt_speech_token_len": flow_prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "llm_embedding": llm_embedding,
            "flow_embedding": flow_embedding,
        }
        return self._generate(inputs, max_speech_tokens)

    @torch.inference_mode()
    def inference_cross_lingual(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        llm_prompt_speech_token: torch.Tensor,
        llm_prompt_speech_token_len: torch.Tensor,
        flow_prompt_speech_token: torch.Tensor,
        flow_prompt_speech_token_len: torch.Tensor,
        prompt_speech_feat: torch.Tensor,
        prompt_speech_feat_len: torch.Tensor,
        llm_embedding: torch.Tensor,
        flow_embedding: torch.Tensor | None = None,
        max_speech_tokens: int = 50,
    ) -> torch.Tensor:
        """Cross-lingual mode: synthesize `text` (any language) in the voice of a
        reference audio clip in a *different* language. The LLM does NOT see the
        prompt text (it would bind to the wrong language); it only sees the
        prompt speech tokens. The flow uses prompt mel + speaker embedding.

        Mirrors the official ``frontend_cross_lingual``.
        """
        if flow_embedding is None:
            flow_embedding = llm_embedding
        inputs = {
            "text": text,
            "text_len": text_len,
            "prompt_text": self._empty_prompt_text(),
            "prompt_text_len": self._empty_prompt_text_len(),
            "llm_prompt_speech_token": llm_prompt_speech_token,
            "llm_prompt_speech_token_len": llm_prompt_speech_token_len,
            "flow_prompt_speech_token": flow_prompt_speech_token,
            "flow_prompt_speech_token_len": flow_prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "llm_embedding": llm_embedding,
            "flow_embedding": flow_embedding,
        }
        return self._generate(inputs, max_speech_tokens)

    @torch.inference_mode()
    def inference_instruct(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        instruct_text: torch.Tensor,
        instruct_text_len: torch.Tensor,
        llm_prompt_speech_token: torch.Tensor,
        llm_prompt_speech_token_len: torch.Tensor,
        flow_prompt_speech_token: torch.Tensor,
        flow_prompt_speech_token_len: torch.Tensor,
        prompt_speech_feat: torch.Tensor,
        prompt_speech_feat_len: torch.Tensor,
        flow_embedding: torch.Tensor,
        max_speech_tokens: int = 50,
    ) -> torch.Tensor:
        """Instruct mode: synthesize expressive speech following an instruction (e.g.
        "say it angrily", "use a Cantonese accent"). The LLM gets the *instruct* text
        as its prompt_text (NOT a transcript of the reference audio). The speaker
        embedding is DROPPED (set to shape (0, 192)) to avoid leaking the reference
        voice — the LLM uses the instruct text to set the style.

        Mirrors the official ``frontend_instruct``: ``del llm_embedding`` is
        expressed by passing an embedding of shape (0, 192), which the LLM's
        ``if embedding.shape[0] != 0`` check uses to skip ``spk_embed_affine_layer``.
        """
        inputs = {
            "text": text,
            "text_len": text_len,
            "prompt_text": instruct_text,
            "prompt_text_len": instruct_text_len,
            "llm_prompt_speech_token": llm_prompt_speech_token,
            "llm_prompt_speech_token_len": llm_prompt_speech_token_len,
            "flow_prompt_speech_token": flow_prompt_speech_token,
            "flow_prompt_speech_token_len": flow_prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "llm_embedding": self._drop_spk_embedding(),
            "flow_embedding": flow_embedding,
        }
        return self._generate(inputs, max_speech_tokens)

    def forward(self, x):
        raise NotImplementedError("Use inference_sft (or other inference_* methods) for generation.")
