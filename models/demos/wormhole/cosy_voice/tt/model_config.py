# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN model configuration for CosyVoice on Wormhole.

Defines memory layouts, sharding strategies, data types, and core grid
configurations for running CosyVoice components on TT hardware.
"""

import ttnn
from models.demos.wormhole.cosy_voice.reference.args import CosyVoiceModelConfig, Qwen2Config


def create_qwen2_model_config(
    qwen2_config: Qwen2Config,
    batch_size: int = 1,
    seq_len: int = 1,
    max_seq_len: int = 2048,
):
    """Create TTNN configuration for the Qwen2-0.5B LLM backbone.

    Args:
        qwen2_config: Qwen2 architecture parameters
        batch_size: Batch size for inference (typically 1 for TTS)
        seq_len: Current sequence length (1 for decode, variable for prefill)
        max_seq_len: Maximum sequence length for KV cache pre-allocation
    """
    configs = {}

    hidden_size = qwen2_config.hidden_size  # 896
    num_heads = qwen2_config.num_attention_heads  # 14
    num_kv_heads = qwen2_config.num_key_value_heads  # 2
    head_dim = qwen2_config.head_dim  # 64
    intermediate_size = qwen2_config.intermediate_size  # 4864

    configs["hidden_size"] = hidden_size
    configs["num_heads"] = num_heads
    configs["num_kv_heads"] = num_kv_heads
    configs["head_dim"] = head_dim
    configs["intermediate_size"] = intermediate_size
    configs["num_layers"] = qwen2_config.num_hidden_layers
    configs["max_seq_len"] = max_seq_len
    configs["batch_size"] = batch_size
    configs["seq_len"] = seq_len
    configs["rms_norm_eps"] = qwen2_config.rms_norm_eps
    configs["rope_theta"] = qwen2_config.rope_theta

    # Data types - BF16 for Stage 1 bring-up
    configs["dtype"] = {
        "activations": ttnn.bfloat16,
        "weights": ttnn.bfloat16,
    }

    # Core grid configuration for Wormhole
    configs["core_grid"] = ttnn.CoreGrid(y=8, x=8)

    # Memory configurations
    configs["dram_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    configs["l1_memcfg"] = ttnn.L1_MEMORY_CONFIG

    return configs


def create_flow_model_config(model_config: CosyVoiceModelConfig):
    """Create TTNN configuration for the Flow Matching decoder."""
    configs = {}
    flow_cfg = model_config.flow

    configs["input_size"] = flow_cfg.input_size
    configs["output_size"] = flow_cfg.output_size
    configs["vocab_size"] = flow_cfg.vocab_size
    configs["n_timesteps"] = flow_cfg.n_timesteps
    configs["spk_embed_dim"] = flow_cfg.spk_embed_dim
    configs["token_mel_ratio"] = flow_cfg.token_mel_ratio
    configs["pre_lookahead_len"] = flow_cfg.pre_lookahead_len

    # Data types
    configs["dtype"] = {
        "activations": ttnn.bfloat16,
        "weights": ttnn.bfloat16,
    }

    configs["core_grid"] = ttnn.CoreGrid(y=8, x=8)
    configs["dram_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    configs["l1_memcfg"] = ttnn.L1_MEMORY_CONFIG

    return configs


def create_hifigan_model_config(model_config: CosyVoiceModelConfig):
    """Create TTNN configuration for the HiFi-GAN vocoder."""
    configs = {}
    hifigan_cfg = model_config.hifigan

    configs["in_channels"] = hifigan_cfg.in_channels
    configs["upsample_rates"] = hifigan_cfg.upsample_rates
    configs["upsample_kernel_sizes"] = hifigan_cfg.upsample_kernel_sizes
    configs["upsample_initial_channel"] = hifigan_cfg.upsample_initial_channel
    configs["resblock_kernel_sizes"] = hifigan_cfg.resblock_kernel_sizes
    configs["resblock_dilation_sizes"] = hifigan_cfg.resblock_dilation_sizes

    # Data types
    configs["dtype"] = {
        "activations": ttnn.bfloat16,
        "weights": ttnn.bfloat16,
    }

    configs["core_grid"] = ttnn.CoreGrid(y=8, x=8)
    configs["dram_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    configs["l1_memcfg"] = ttnn.L1_MEMORY_CONFIG

    return configs


def create_cosyvoice_model_config(
    model_config: CosyVoiceModelConfig = None,
    batch_size: int = 1,
    max_seq_len: int = 2048,
):
    """Create full TTNN configuration for the CosyVoice pipeline.

    Returns a dict with sub-configs for each pipeline stage.
    """
    if model_config is None:
        model_config = CosyVoiceModelConfig()

    return {
        "llm": create_qwen2_model_config(
            model_config.llm.qwen2,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        ),
        "flow": create_flow_model_config(model_config),
        "hifigan": create_hifigan_model_config(model_config),
        "model_config": model_config,
    }
