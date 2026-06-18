# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CosyVoice-300M E2E demo on TTNN.

Stage-1 bring-up: implements SFT-mode (text -> wav with a predefined speaker
embedding). Other modes (zero-shot, cross-lingual, instruct) will be added
in follow-ups (see HANDOFF.md Phase 5).

Usage::

    # pytest (runs SFT mode on the canned sft_en inputs):
    pytest -svv models/demos/wormhole/cosyvoice/demo/demo.py

    # CLI (writes a wav to ./out.wav):
    python -m models.demos.wormhole.cosyvoice.demo.demo \
        --model-dir pretrained_models/CosyVoice-300M \
        --text "Hello world, this is a test of English synthesis." \
        --output out.wav
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import ttnn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice"))
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "CosyVoice" / "third_party" / "Matcha-TTS"))

from models.demos.wormhole.cosyvoice.reference import golden_prompts
from models.demos.wormhole.cosyvoice.reference.golden_pipeline import _encode_text
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.model import TtCosyVoiceModel
from models.demos.wormhole.cosyvoice.tt.model_config import create_model_config

MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", str(REPO_ROOT / "pretrained_models" / "CosyVoice-300M"))


def _wav_to_int16(wav: torch.Tensor) -> np.ndarray:
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    wav = wav.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
    return (wav * 32767.0).astype(np.int16)


def _save_wav(path: Path, wav_int16: np.ndarray, sample_rate: int = TtCosyVoiceModel.SAMPLE_RATE) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(wav_int16.tobytes())


def _build_tt_model(device):
    """Load reference + construct the TT E2E model."""
    print(f"[demo] Loading reference model from {MODEL_DIR} ...")
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_model.llm.eval()
    ref_model.flow.eval()
    ref_model.hifigan.eval()
    config = create_model_config(batch_size=1, hidden_size=1024)
    tt_model = TtCosyVoiceModel(
        device,
        config,
        args=None,
        ref_model=ref_model,
    )
    return ref_model, tt_model


def _seeded_embedding(seed: int, dim: int = 192) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, dim, generator=g, dtype=torch.float32)


def run_sft_demo(
    device,
    text: str,
    output_path: Path,
    spk_embedding: torch.Tensor,
    max_speech_tokens: int = 50,
):
    """Run SFT-mode TTS. Returns the wav tensor and writes it to `output_path`."""
    ref_model, tt_model = _build_tt_model(device)

    # Reseed before each call so the HiFi-GAN m_source RNG is reproducible
    torch.manual_seed(0)

    text_token = _encode_text(text)
    text_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)

    print(
        f"[demo] Running TT SFT inference: text={text!r} "
        f"({text_token.shape[1]} tokens), spk_embed shape={tuple(spk_embedding.shape)}"
    )
    wav = tt_model.inference_sft(
        text=text_token,
        text_len=text_len,
        llm_embedding=spk_embedding,
        max_speech_tokens=max_speech_tokens,
    )
    print(f"[demo] wav shape={tuple(wav.shape)} duration={wav.shape[-1] / TtCosyVoiceModel.SAMPLE_RATE:.2f}s")
    print(f"[demo] wav stats: min={wav.min():.3f} max={wav.max():.3f} " f"mean={wav.mean():.3f} std={wav.std():.3f}")

    _save_wav(output_path, _wav_to_int16(wav))
    print(f"[demo] Wrote {output_path}")
    return wav


@pytest.mark.parametrize("mesh_device", (1,), indirect=True)
def test_demo_sft(mesh_device, tmp_path):
    """Pytest entry: run SFT on the canned sft_en inputs, save the wav to tmp_path."""
    wav = run_sft_demo(
        device=mesh_device,
        text=golden_prompts.text_for("sft", "en"),
        output_path=tmp_path / "sft_en.wav",
        spk_embedding=_seeded_embedding(seed=1001),
    )
    assert wav.dim() in (1, 2)
    assert torch.isfinite(wav).all()
    assert wav.abs().max() <= 0.99 + 1e-3, f"wav out of audio_limit: {wav.abs().max()}"
    # ~1s of audio for 50 tokens (sample rate 22050)
    assert wav.shape[-1] > 1000, f"wav too short: {wav.shape[-1]} samples"


def main():
    global MODEL_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--text", default=golden_prompts.text_for("sft", "en"))
    parser.add_argument("--output", default="out.wav")
    parser.add_argument("--max-speech-tokens", type=int, default=50)
    parser.add_argument("--spk-seed", type=int, default=1001)
    args = parser.parse_args()

    MODEL_DIR = args.model_dir

    # Open a TT device (same as conftest)
    device = ttnn.open_device(device_id=0, l1_small_size=64 << 10, trace_region_size=128 << 20)
    device.enable_program_cache()
    try:
        spk_emb = _seeded_embedding(seed=args.spk_seed)
        run_sft_demo(
            device=device,
            text=args.text,
            output_path=Path(args.output),
            spk_embedding=spk_emb,
            max_speech_tokens=args.max_speech_tokens,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
