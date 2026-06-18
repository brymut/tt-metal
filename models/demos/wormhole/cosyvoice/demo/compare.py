# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CosyVoice-300M SFT listen-and-compare demo on TTNN.

Generates speech for a predefined speaker from text, writing BOTH the TT
(on-device) output and the reference (CPU PyTorch) output as separate wavs so
you can A/B them by ear.

SFT mode (predefined speaker) is the simplest mode: no reference audio is
needed — the speaker identity comes entirely from the speaker embedding stored
in ``spk2info.pt``. Per the official ``frontend_sft`` (cosyvoice/cli/frontend.py),
SFT feeds the model ONLY ``text + embedding``; all prompt tokens / prompt mel
are left empty (the reference ``model.tts`` defaults them to empty tensors).
Both the TT and reference paths here use that same official-correct input set,
so the A/B is apples-to-apples and the only intended difference is the TTNN
numerical path vs the PyTorch reference.

Usage::

    # pytest (canned English text, writes both wavs to tmp_path)
    pytest -svv models/demos/wormhole/cosyvoice/demo/compare.py

    # CLI (writes tt.wav and ref.wav into --output-dir)
    python -m models.demos.wormhole.cosyvoice.demo.compare \\
        --text "Hello world, this is a test of English synthesis." \\
        --speaker en_speaker_6s \\
        --output-dir ./out_sft

    # Non-English (prepend the language tag)
    python -m models.demos.wormhole.cosyvoice.demo.compare \\
        --text "<|zh|>今天天气真好，我们一起去公园散步吧。" \\
        --speaker zh_speaker_3s \\
        --output-dir ./out_sft

Available speakers (from ``pretrained_models/CosyVoice-300M/spk2info.pt``)::

    en_speaker_6s       (English,    5.86s LibriSpeech)
    zh_speaker_3s       (Chinese,    3.48s CosyVoice v1 asset)
    ja_speaker_11s      (Japanese,  11.10s FLEURS)
    ko_speaker_11s      (Korean,    10.56s FLEURS)
    yue_speaker_15s     (Cantonese, 15.48s FLEURS)
    xling_speaker_14s   (Multilingual, 13.75s CosyVoice v1 asset)
"""

import argparse
import os
import sys
import wave
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

from models.demos.wormhole.cosyvoice.reference.golden_pipeline import (  # noqa: E402
    MAX_DECODED_SPEECH_TOKENS,
    SAMPLE_RATE,
    _encode_text,
    _get_predefined_speakers,
    run_reference_pipeline,
)
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel  # noqa: E402
from models.demos.wormhole.cosyvoice.tt.model import TtCosyVoiceModel  # noqa: E402
from models.demos.wormhole.cosyvoice.tt.model_config import create_model_config  # noqa: E402

MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", str(REPO_ROOT / "pretrained_models" / "CosyVoice-300M"))

DEFAULT_SPEAKER = "en_speaker_6s"
DEFAULT_MAX_SPEECH_TOKENS = MAX_DECODED_SPEECH_TOKENS  # ~4s of audio
DEFAULT_SEED = 0


def _wav_to_int16(wav: torch.Tensor) -> np.ndarray:
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    wav = wav.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
    return (wav * 32767.0).astype(np.int16)


def _save_wav(path: Path, wav_int16: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(wav_int16.tobytes())


def _build_models(device):
    """Load reference + construct the TT E2E model."""
    print(f"[compare] Loading reference model from {MODEL_DIR} ...")
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_model.llm.eval()
    ref_model.flow.eval()
    ref_model.hifigan.eval()
    config = create_model_config(batch_size=1, hidden_size=1024)
    tt_model = TtCosyVoiceModel(device, config, args=None, ref_model=ref_model)
    return ref_model, tt_model


def _resolve_speaker_embedding(speaker_id: str) -> torch.Tensor:
    """Return the (1, 192) llm_embedding for a predefined speaker.

    For SFT mode the official ``frontend_sft`` passes a single ``embedding``
    used as both the LLM and the flow speaker embedding. ``spk2info.pt`` stores
    separate ``llm_embedding`` / ``flow_embedding``; we use ``llm_embedding``
    (the TT ``inference_sft`` reuses it as the flow embedding internally).
    """
    spk2info = _get_predefined_speakers()
    if speaker_id not in spk2info:
        available = ", ".join(sorted(spk2info.keys()))
        raise ValueError(f"unknown speaker {speaker_id!r}. Available: {available}")
    return spk2info[speaker_id]["llm_embedding"].clone()


def _build_sft_inputs(text: str, llm_embedding: torch.Tensor) -> dict:
    """Build the OFFICIAL-correct SFT input dict.

    Mirrors ``frontend_sft``: only text + embedding are meaningful; all prompt
    tensors are empty (matching the reference ``model.tts`` defaults). Feeding
    the same dict to both the reference pipeline and the TT path keeps the A/B
    apples-to-apples.
    """
    text_token = _encode_text(text)
    text_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)
    return {
        "text": text_token,
        "text_len": text_len,
        "prompt_text": torch.zeros(1, 0, dtype=torch.int32),
        "prompt_text_len": torch.tensor([0], dtype=torch.int32),
        "llm_prompt_speech_token": torch.zeros(1, 0, dtype=torch.int32),
        "llm_prompt_speech_token_len": torch.tensor([0], dtype=torch.int32),
        "flow_prompt_speech_token": torch.zeros(1, 0, dtype=torch.int32),
        "flow_prompt_speech_token_len": torch.tensor([0], dtype=torch.int32),
        "prompt_speech_feat": torch.zeros(1, 0, 80, dtype=torch.float32),
        "prompt_speech_feat_len": torch.tensor([0], dtype=torch.int32),
        "llm_embedding": llm_embedding,
        "flow_embedding": llm_embedding,
    }


def _print_wav_stats(label: str, wav: torch.Tensor) -> None:
    dur = wav.shape[-1] / SAMPLE_RATE
    print(
        f"[compare] {label}: shape={tuple(wav.shape)} duration={dur:.2f}s "
        f"min={wav.min():.3f} max={wav.max():.3f} mean={wav.mean():.3f} std={wav.std():.3f}"
    )


def run_sft_compare(
    device,
    text: str,
    speaker_id: str,
    output_dir: Path,
    max_speech_tokens: int = DEFAULT_MAX_SPEECH_TOKENS,
    seed: int = DEFAULT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SFT on both TT and reference, write both wavs, return (tt_wav, ref_wav)."""
    ref_model, tt_model = _build_models(device)
    llm_embedding = _resolve_speaker_embedding(speaker_id)
    inputs = _build_sft_inputs(text, llm_embedding)

    print(
        f"[compare] SFT mode: text={text!r} ({inputs['text'].shape[1]} tokens), "
        f"speaker={speaker_id!r}, max_speech_tokens={max_speech_tokens}, seed={seed}"
    )

    # ---- Reference (CPU PyTorch) ----
    # Seed the global RNG identically before both paths so the LLM top-k sampling
    # draws the same nominal tokens; the only intended difference is the TTNN
    # numerical path. (HiFi-GAN also uses the global RNG for its noise source,
    # so even ref-vs-ref wavs aren't bit-exact — that's fine for an ear A/B.)
    torch.manual_seed(seed)
    print("[compare] Running reference pipeline (CPU) ...")
    ref_wav, ref_mel, ref_tokens = run_reference_pipeline(
        ref_model, inputs, max_speech_tokens=max_speech_tokens, sampling_seed=seed
    )
    _print_wav_stats("ref", ref_wav)

    # ---- TT (on-device) ----
    torch.manual_seed(seed)
    print("[compare] Running TT pipeline (N300) ...")
    tt_wav = tt_model.inference_sft(
        text=inputs["text"],
        text_len=inputs["text_len"],
        llm_embedding=llm_embedding,
        max_speech_tokens=max_speech_tokens,
    )
    _print_wav_stats("tt ", tt_wav)

    # ---- Save both wavs ----
    tt_path = output_dir / "tt.wav"
    ref_path = output_dir / "ref.wav"
    _save_wav(tt_path, _wav_to_int16(tt_wav))
    _save_wav(ref_path, _wav_to_int16(ref_wav))
    print(f"[compare] Wrote TT  wav -> {tt_path}")
    print(f"[compare] Wrote REF wav -> {ref_path}")
    print("[compare] Listen to both and A/B by ear. Reference tokens: " f"{len(ref_tokens)} -> {ref_mel.shape}")
    return tt_wav, ref_wav


@pytest.mark.parametrize("mesh_device", (1,), indirect=True)
def test_compare_sft(mesh_device, tmp_path):
    """Pytest entry: run SFT A/B on canned English text, write both wavs."""
    tt_wav, ref_wav = run_sft_compare(
        device=mesh_device,
        text="Hello world, this is a test of English synthesis.",
        speaker_id=DEFAULT_SPEAKER,
        output_dir=tmp_path,
    )
    for label, w in (("tt", tt_wav), ("ref", ref_wav)):
        assert torch.isfinite(w).all(), f"{label} wav has non-finite values"
        assert w.abs().max() <= 0.99 + 1e-3, f"{label} wav out of range: {w.abs().max()}"
        assert w.shape[-1] > 1000, f"{label} wav too short: {w.shape[-1]}"


def main():
    global MODEL_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument(
        "--text",
        required=True,
        help="TTS text to synthesize. Prepend a language tag (e.g. '<|zh|>') for non-English.",
    )
    parser.add_argument(
        "--speaker",
        default=DEFAULT_SPEAKER,
        help="Predefined speaker ID (see spk2info.pt). Default: en_speaker_6s.",
    )
    parser.add_argument("--output-dir", default="./out_sft")
    parser.add_argument("--max-speech-tokens", type=int, default=DEFAULT_MAX_SPEECH_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    MODEL_DIR = args.model_dir

    device = ttnn.open_device(device_id=0, l1_small_size=64 << 10, trace_region_size=128 << 20)
    device.enable_program_cache()
    try:
        run_sft_compare(
            device=device,
            text=args.text,
            speaker_id=args.speaker,
            output_dir=Path(args.output_dir),
            max_speech_tokens=args.max_speech_tokens,
            seed=args.seed,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
