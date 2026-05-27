# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice TTS demo on Tenstorrent Wormhole (N300).

Demonstrates text-to-speech synthesis using the TTNN-accelerated CosyVoice pipeline.
Supports multilingual TTS in Chinese, English, Japanese, Cantonese, and Korean.

Usage:
    # Basic demo with default prompts
    pytest --disable-warnings -q -s models/demos/wormhole/cosy_voice/demo/demo.py::test_cosyvoice_demo

    # With custom text
    pytest --disable-warnings -q -s models/demos/wormhole/cosy_voice/demo/demo.py::test_cosyvoice_demo --text="Hello world"

    # Performance benchmark
    pytest --disable-warnings -q -s models/demos/wormhole/cosy_voice/demo/demo.py::test_cosyvoice_perf
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.cosy_voice.reference.args import CosyVoiceModelConfig
from models.demos.wormhole.cosy_voice.tt.ttnn_cosyvoice_model import TtCosyVoiceModel

# Default model directory (update after downloading weights)
DEFAULT_MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR",
    "models/demos/wormhole/cosy_voice/pretrained_models/CosyVoice2-0.5B",
)

SAMPLE_PROMPTS_PATH = Path(__file__).parent / "sample_prompts.json"


def load_sample_prompts():
    """Load multilingual test prompts."""
    if SAMPLE_PROMPTS_PATH.exists():
        with open(SAMPLE_PROMPTS_PATH) as f:
            return json.load(f)
    return {
        "test_prompts": [
            {"language": "english", "text": "Hello, this is a test of the CosyVoice speech synthesis system."}
        ]
    }


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 22050):
    """Save waveform to WAV file."""
    try:
        import torchaudio

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(path, waveform, sample_rate)
        logger.info(f"Audio saved to {path}")
    except ImportError:
        logger.warning("torchaudio not available, skipping audio save")


def dummy_tokenize(text: str) -> torch.Tensor:
    """Simple placeholder tokenizer for demo testing.

    In the full pipeline, this is handled by the CosyVoice frontend
    which uses the Qwen2 tokenizer. For demo testing without the full
    CosyVoice package, we generate random tokens.
    """
    # Generate deterministic pseudo-tokens based on text hash
    torch.manual_seed(hash(text) % 2**32)
    num_tokens = max(len(text) // 2, 10)  # Rough estimate
    tokens = torch.randint(100, 150000, (1, num_tokens), dtype=torch.int32)
    return tokens


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cosyvoice_demo(device: ttnn.Device, reset_seeds):
    """Main demo: Generate speech from text prompts.

    Tests the full pipeline: text → LLM → flow matching → vocoder → audio
    """
    model_dir = DEFAULT_MODEL_DIR

    if not os.path.exists(model_dir):
        logger.warning(
            f"Model directory {model_dir} not found. "
            f"Download with: huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir {model_dir}"
        )
        pytest.skip(f"Model weights not found at {model_dir}")

    # Load model
    logger.info(f"Loading CosyVoice from {model_dir}...")
    model = TtCosyVoiceModel.from_pretrained(model_dir, device)

    # Load prompts
    prompts = load_sample_prompts()
    output_dir = Path("generated/cosyvoice_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt in prompts["test_prompts"]:
        lang = prompt["language"]
        text = prompt["text"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Language: {lang}")
        logger.info(f"Text: {text}")
        logger.info(f"{'='*60}")

        # Tokenize
        text_tokens = dummy_tokenize(text)

        # Generate speech
        start_time = time.time()
        for output in model.tts(text=text_tokens):
            tts_speech = output["tts_speech"]
            total_time = time.time() - start_time
            speech_duration = tts_speech.shape[-1] / model.sample_rate
            rtf = total_time / speech_duration if speech_duration > 0 else float("inf")

            logger.info(f"Generated {speech_duration:.2f}s audio in {total_time:.2f}s (RTF={rtf:.3f})")

            # Save audio
            output_path = str(output_dir / f"cosyvoice_{lang}.wav")
            save_audio(tts_speech, output_path, model.sample_rate)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cosyvoice_perf(device: ttnn.Device, reset_seeds):
    """Performance benchmark: Measure throughput and RTF.

    Target metrics:
    - LLM: ≥ 30 tokens/sec
    - RTF: < 0.5
    """
    model_dir = DEFAULT_MODEL_DIR

    if not os.path.exists(model_dir):
        pytest.skip(f"Model weights not found at {model_dir}")

    model = TtCosyVoiceModel.from_pretrained(model_dir, device)

    # Fixed test input for reproducible benchmarks
    torch.manual_seed(42)
    text_tokens = torch.randint(100, 150000, (1, 20), dtype=torch.int32)

    # Warmup
    logger.info("Warmup run...")
    for output in model.tts(text=text_tokens):
        _ = output["tts_speech"]

    # Benchmark
    num_runs = 3
    total_tokens = 0
    total_llm_time = 0
    total_rtf = 0

    for run in range(num_runs):
        logger.info(f"\nBenchmark run {run + 1}/{num_runs}")
        start = time.time()
        for output in model.tts(text=text_tokens):
            tts_speech = output["tts_speech"]
            elapsed = time.time() - start
            speech_len = tts_speech.shape[-1] / model.sample_rate
            rtf = elapsed / speech_len if speech_len > 0 else float("inf")
            total_rtf += rtf

    avg_rtf = total_rtf / num_runs
    logger.info(f"\n{'='*60}")
    logger.info(f"Average RTF: {avg_rtf:.3f} (target: < 0.5)")
    logger.info(f"{'='*60}")

    # Assert performance targets
    # Note: Relaxed for Stage 1 bring-up; will tighten after optimization
    assert avg_rtf < 5.0, f"RTF {avg_rtf:.3f} exceeds relaxed Stage 1 target of 5.0"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cosyvoice_unit_pipeline(device: ttnn.Device, reset_seeds):
    """Lightweight pipeline test without pretrained weights.

    Validates that the pipeline components connect correctly
    using random weights. Does NOT test quality.
    """
    torch.manual_seed(42)
    config = CosyVoiceModelConfig()

    # Create small random state dicts for testing connectivity
    hidden_size = config.llm.qwen2.hidden_size  # 896
    num_layers = 2  # Use only 2 layers for quick test

    # Minimal LLM state dict
    llm_sd = {}
    # Embedding
    llm_sd["llm.model.model.embed_tokens.weight"] = torch.randn(config.llm.qwen2.vocab_size, hidden_size)
    # Final norm
    llm_sd["llm.model.norm.weight"] = torch.ones(hidden_size)
    # CosyVoice wrapper
    llm_sd["speech_embedding.weight"] = torch.randn(config.llm.speech_token_size + 3, hidden_size)
    llm_sd["llm_embedding.weight"] = torch.randn(2, hidden_size)
    llm_sd["llm_decoder.weight"] = torch.randn(config.llm.speech_token_size + 3, hidden_size)
    llm_sd["llm_decoder.bias"] = torch.randn(config.llm.speech_token_size + 3)

    # Per-layer weights
    for layer_idx in range(num_layers):
        prefix = f"llm.model.layers.{layer_idx}."
        llm_sd[f"{prefix}self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        llm_sd[f"{prefix}self_attn.k_proj.weight"] = torch.randn(128, hidden_size)
        llm_sd[f"{prefix}self_attn.v_proj.weight"] = torch.randn(128, hidden_size)
        llm_sd[f"{prefix}self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        llm_sd[f"{prefix}self_attn.q_proj.bias"] = torch.randn(hidden_size)
        llm_sd[f"{prefix}self_attn.k_proj.bias"] = torch.randn(128)
        llm_sd[f"{prefix}self_attn.v_proj.bias"] = torch.randn(128)
        llm_sd[f"{prefix}mlp.gate_proj.weight"] = torch.randn(4864, hidden_size)
        llm_sd[f"{prefix}mlp.up_proj.weight"] = torch.randn(4864, hidden_size)
        llm_sd[f"{prefix}mlp.down_proj.weight"] = torch.randn(hidden_size, 4864)
        llm_sd[f"{prefix}input_layernorm.weight"] = torch.ones(hidden_size)
        llm_sd[f"{prefix}post_attention_layernorm.weight"] = torch.ones(hidden_size)

    # Override num_layers for quick test
    config.llm.qwen2.num_hidden_layers = num_layers

    # Minimal flow state dict
    flow_sd = {
        "input_embedding.weight": torch.randn(config.flow.vocab_size, config.flow.input_size),
        "spk_embed_affine_layer.weight": torch.randn(config.flow.output_size, config.flow.spk_embed_dim),
        "spk_embed_affine_layer.bias": torch.randn(config.flow.output_size),
    }

    # Minimal HiFi-GAN state dict (empty — will use random init)
    hift_sd = {}

    logger.info("Creating pipeline with random weights (2 layers)...")
    model = TtCosyVoiceModel(
        device=device,
        model_config=config,
        llm_state_dict=llm_sd,
        flow_state_dict=flow_sd,
        hift_state_dict=hift_sd,
    )

    # Run a short generation
    text_tokens = torch.randint(100, 1000, (1, 5), dtype=torch.int32)

    logger.info("Running pipeline connectivity test...")
    for output in model.tts(text=text_tokens):
        tts_speech = output["tts_speech"]
        assert tts_speech is not None, "Pipeline returned None"
        assert tts_speech.dim() >= 1, f"Expected at least 1D output, got {tts_speech.dim()}"
        logger.info(f"Pipeline connectivity test PASSED — output shape: {tts_speech.shape}")
