# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice model arguments and configuration.

Defines the model architecture parameters for CosyVoice2 (Qwen2-0.5B backbone)
and the pipeline configuration for LLM, Flow Matching, and HiFi-GAN components.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List


class ModelMode(IntEnum):
    DECODE = 0
    PREFILL = 1


class InferenceMode(IntEnum):
    SFT = 0
    ZERO_SHOT = 1
    CROSS_LINGUAL = 2
    INSTRUCT = 3
    VOICE_CONVERSION = 4


@dataclass
class Qwen2Config:
    """Configuration for the Qwen2-0.5B LLM backbone."""

    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2  # GQA: 2 KV heads
    head_dim: int = 64
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    use_sliding_window: bool = False
    sliding_window: int = 32768
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True


@dataclass
class CosyVoiceLLMConfig:
    """Configuration for the CosyVoice LLM wrapper around Qwen2."""

    llm_input_size: int = 896
    llm_output_size: int = 896
    speech_token_size: int = 6561  # FSQ vocabulary size for CosyVoice2
    # Special tokens
    sos_token: int = 0
    task_id_token: int = 1
    eos_token: int = 6561  # speech_token_size
    fill_token: int = 6563  # speech_token_size + 2
    # Sampling
    default_sampling: int = 25  # top-k
    max_token_text_ratio: float = 20.0
    min_token_text_ratio: float = 2.0
    # Bi-streaming mix ratio
    mix_ratio: List[int] = field(default_factory=lambda: [5, 15])
    # Qwen2 backbone config
    qwen2: Qwen2Config = field(default_factory=Qwen2Config)


@dataclass
class FlowMatchingConfig:
    """Configuration for the Flow Matching (CFM) decoder."""

    input_size: int = 512
    output_size: int = 80  # mel spectrogram dimensions
    spk_embed_dim: int = 192
    vocab_size: int = 6561  # speech token vocabulary
    input_frame_rate: int = 25  # 25 Hz token rate
    token_mel_ratio: int = 2  # each token maps to 2 mel frames
    pre_lookahead_len: int = 3
    # CFM solver
    n_timesteps: int = 10  # Euler solver steps
    sigma_min: float = 1e-6
    t_scheduler: str = "cosine"
    inference_cfg_rate: float = 0.7  # classifier-free guidance rate
    # Encoder
    encoder_output_size: int = 512
    # DiT estimator
    dit_channels: List[int] = field(default_factory=lambda: [256, 256])
    dit_n_blocks: int = 4
    dit_num_mid_blocks: int = 12
    dit_num_heads: int = 8
    dit_attention_head_dim: int = 64


@dataclass
class HiFiGANConfig:
    """Configuration for the HiFi-GAN vocoder."""

    in_channels: int = 80  # mel spectrogram channels
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    sample_rate: int = 22050


@dataclass
class CosyVoiceModelConfig:
    """Top-level configuration for the full CosyVoice pipeline."""

    model_version: str = "CosyVoice2-0.5B"
    sample_rate: int = 22050
    # Component configs
    llm: CosyVoiceLLMConfig = field(default_factory=CosyVoiceLLMConfig)
    flow: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    hifigan: HiFiGANConfig = field(default_factory=HiFiGANConfig)
    # Streaming
    token_hop_len: int = 25
    token_max_hop_len: int = 100
    stream_scale_factor: int = 2
    # Memory cache
    mel_cache_len: int = 8
