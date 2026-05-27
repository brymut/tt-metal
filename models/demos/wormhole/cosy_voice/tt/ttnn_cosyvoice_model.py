# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the full CosyVoice TTS pipeline.

Orchestrates the 3-stage pipeline:
    Text → LLM (semantic tokens) → Flow Matching (mel) → HiFi-GAN (audio)

Supports all inference modes: SFT, Zero-shot, Cross-lingual, Instruct.
The mode-specific preprocessing happens on CPU; the TTNN pipeline is mode-agnostic.
"""

import threading
import time
from typing import Dict, Generator, Optional

import torch
from loguru import logger

import ttnn
from models.demos.wormhole.cosy_voice.reference.args import CosyVoiceModelConfig
from models.demos.wormhole.cosy_voice.tt.model_config import create_cosyvoice_model_config
from models.demos.wormhole.cosy_voice.tt.ttnn_cosyvoice_llm import TtCosyVoiceLLM
from models.demos.wormhole.cosy_voice.tt.ttnn_flow_matching import TtFlowMatching
from models.demos.wormhole.cosy_voice.tt.ttnn_hifigan import TtHiFiGAN


class TtCosyVoiceModel:
    """Full CosyVoice TTS pipeline on Tenstorrent hardware.

    Device split:
    - LLM backbone: TTNN on Wormhole (autoregressive decode)
    - Flow Matching: TTNN on Wormhole (CFM Euler solver) [Stage 1: CPU fallback]
    - HiFi-GAN: CPU fallback (ConvTranspose1d not supported in TTNN)
    - Text frontend, speaker encoder, speech tokenizer: CPU (ONNX)
    """

    def __init__(
        self,
        device: ttnn.Device,
        model_config: CosyVoiceModelConfig = None,
        llm_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        flow_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        hift_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        max_seq_len: int = 2048,
    ):
        """Initialize the full CosyVoice TTNN pipeline.

        Args:
            device: TTNN device (Wormhole N300)
            model_config: Model architecture configuration
            llm_state_dict: LLM weights from llm.pt
            flow_state_dict: Flow matching weights from flow.pt
            hift_state_dict: HiFi-GAN weights from hift.pt
            max_seq_len: Maximum sequence length for KV cache
        """
        self.device = device

        if model_config is None:
            model_config = CosyVoiceModelConfig()
        self.model_config = model_config

        # Create TTNN configs
        configs = create_cosyvoice_model_config(model_config, max_seq_len=max_seq_len)

        # Streaming parameters
        self.token_hop_len = model_config.token_hop_len  # 25
        self.token_max_hop_len = model_config.token_max_hop_len  # 100
        self.stream_scale_factor = model_config.stream_scale_factor  # 2
        self.sample_rate = model_config.sample_rate

        # Initialize LLM backbone
        if llm_state_dict is not None:
            logger.info("Initializing TTNN LLM backbone (Qwen2-0.5B)...")
            self.llm = TtCosyVoiceLLM(
                device=device,
                configs=configs["llm"],
                llm_config=model_config.llm,
                llm_state_dict=llm_state_dict,
            )
        else:
            self.llm = None
            logger.warning("LLM state dict not provided, LLM will not be available")

        # Initialize Flow Matching decoder
        if flow_state_dict is not None:
            logger.info("Initializing Flow Matching decoder...")
            self.flow = TtFlowMatching(
                device=device,
                configs=configs["flow"],
                flow_state_dict=flow_state_dict,
            )
        else:
            self.flow = None
            logger.warning("Flow state dict not provided, Flow decoder will not be available")

        # Initialize HiFi-GAN vocoder (CPU fallback)
        if hift_state_dict is not None:
            logger.info("Initializing HiFi-GAN vocoder (CPU)...")
            self.hift = TtHiFiGAN(
                configs=configs["hifigan"],
                hift_state_dict=hift_state_dict,
            )
        else:
            self.hift = None
            logger.warning("HiFi-GAN state dict not provided, vocoder will not be available")

        # Thread safety
        self.lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: ttnn.Device,
        max_seq_len: int = 2048,
    ) -> "TtCosyVoiceModel":
        """Load a pretrained CosyVoice model from directory.

        Expected files:
            model_dir/llm.pt
            model_dir/flow.pt
            model_dir/hift.pt

        Args:
            model_dir: Path to pretrained model directory
            device: TTNN device
            max_seq_len: Maximum sequence length

        Returns:
            Initialized TtCosyVoiceModel
        """
        import os

        llm_path = os.path.join(model_dir, "llm.pt")
        flow_path = os.path.join(model_dir, "flow.pt")
        hift_path = os.path.join(model_dir, "hift.pt")

        llm_sd = torch.load(llm_path, map_location="cpu", weights_only=True) if os.path.exists(llm_path) else None
        flow_sd = torch.load(flow_path, map_location="cpu", weights_only=True) if os.path.exists(flow_path) else None
        hift_sd = torch.load(hift_path, map_location="cpu", weights_only=True) if os.path.exists(hift_path) else None

        return cls(
            device=device,
            llm_state_dict=llm_sd,
            flow_state_dict=flow_sd,
            hift_state_dict=hift_sd,
            max_seq_len=max_seq_len,
        )

    def tts(
        self,
        text: torch.Tensor,
        prompt_text: torch.Tensor = None,
        llm_prompt_speech_token: torch.Tensor = None,
        flow_prompt_speech_token: torch.Tensor = None,
        prompt_speech_feat: torch.Tensor = None,
        flow_embedding: torch.Tensor = None,
        llm_embedding: torch.Tensor = None,
        stream: bool = False,
        speed: float = 1.0,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Full text-to-speech pipeline.

        This is the main entry point matching CosyVoice's model.tts() interface.

        Args:
            text: Input text token IDs (1, text_len)
            prompt_text: Prompt text tokens for zero-shot (1, prompt_len)
            llm_prompt_speech_token: Prompt speech tokens for LLM (1, N)
            flow_prompt_speech_token: Prompt speech tokens for flow (1, N)
            prompt_speech_feat: Prompt mel features (1, T, 80)
            flow_embedding: Speaker embedding for flow (1, 192)
            llm_embedding: Speaker embedding for LLM (1, 192)
            stream: Whether to stream output chunks
            speed: Speech speed factor (1.0 = normal)

        Yields:
            Dict with 'tts_speech' key containing audio waveform tensor
        """
        # Default values
        if prompt_text is None:
            prompt_text = torch.zeros(1, 0, dtype=torch.int32)
        if llm_prompt_speech_token is None:
            llm_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        if flow_prompt_speech_token is None:
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        if prompt_speech_feat is None:
            prompt_speech_feat = torch.zeros(1, 0, 80)
        if flow_embedding is None:
            flow_embedding = torch.zeros(1, 192)
        if llm_embedding is None:
            llm_embedding = torch.zeros(1, 192)

        assert self.llm is not None, "LLM not initialized"
        assert self.flow is not None, "Flow decoder not initialized"
        assert self.hift is not None, "HiFi-GAN vocoder not initialized"

        # 1. LLM: Generate semantic tokens
        logger.info("Generating semantic tokens via LLM...")
        start_time = time.time()

        all_tokens = []
        for token_id in self.llm.inference(
            text=text,
            text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
            prompt_text=prompt_text,
            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
            prompt_speech_token=llm_prompt_speech_token,
            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32),
            embedding=llm_embedding,
        ):
            all_tokens.append(token_id)

        llm_time = time.time() - start_time
        tokens_per_sec = len(all_tokens) / llm_time if llm_time > 0 else 0
        logger.info(f"LLM generated {len(all_tokens)} tokens in {llm_time:.2f}s " f"({tokens_per_sec:.1f} tokens/sec)")

        if len(all_tokens) == 0:
            logger.warning("LLM generated 0 tokens, yielding silence")
            yield {"tts_speech": torch.zeros(1, self.sample_rate)}
            return

        # 2. Flow Matching: Tokens → Mel spectrogram
        logger.info("Running flow matching decoder...")
        start_time = time.time()

        speech_tokens = torch.tensor(all_tokens).unsqueeze(0)
        tts_mel, _ = self.flow.inference(
            token=speech_tokens,
            token_len=torch.tensor([speech_tokens.shape[1]], dtype=torch.int32),
            prompt_token=flow_prompt_speech_token,
            prompt_token_len=torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_speech_feat,
            prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32),
            embedding=flow_embedding,
            finalize=True,
        )

        flow_time = time.time() - start_time
        logger.info(f"Flow matching completed in {flow_time:.2f}s")

        # 3. HiFi-GAN: Mel → Audio waveform
        logger.info("Running HiFi-GAN vocoder...")
        start_time = time.time()

        if speed != 1.0:
            tts_mel = torch.nn.functional.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear")

        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)

        vocoder_time = time.time() - start_time
        speech_len = tts_speech.shape[-1] / self.sample_rate
        total_time = llm_time + flow_time + vocoder_time
        rtf = total_time / speech_len if speech_len > 0 else float("inf")

        logger.info(
            f"Vocoder completed in {vocoder_time:.2f}s | "
            f"Audio: {speech_len:.2f}s | RTF: {rtf:.3f} | "
            f"Total: {total_time:.2f}s"
        )

        yield {"tts_speech": tts_speech.cpu()}
