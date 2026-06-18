# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CosyVoice-300M instruct-mode TTS demo on TTNN.

Synthesizes speech with an instruction (e.g. "speak in a calm tone")
in the voice of a predefined speaker. The user manually prepends the
language tag to the text and instruction (e.g. ``<|zh|>你好世界`` for
Chinese). No auto language detection.

Usage::

    # pytest (runs on the canned English text/instruction pair)
    pytest -svv models/demos/wormhole/cosyvoice/demo/tts_instruct.py

    # CLI (English, default speaker)
    python -m models.demos.wormhole.cosyvoice.demo.tts_instruct \\
        --text "Hello world, this is a test of English synthesis." \\
        --instruction "Speak in a calm and friendly tone." \\
        --output out.wav

    # Non-English (user prepends the language tag)
    python -m models.demos.wormhole.cosyvoice.demo.tts_instruct \\
        --text "<|zh|>今天天气真好, 我们一起去公园散步吧。" \\
        --instruction "<|zh|>请用温柔舒缓的语气朗读。" \\
        --speaker zh_speaker_3s \\
        --output out.wav

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
    _encode_text,
    _get_predefined_speakers,
)
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel  # noqa: E402
from models.demos.wormhole.cosyvoice.tt.model import TtCosyVoiceModel  # noqa: E402
from models.demos.wormhole.cosyvoice.tt.model_config import create_model_config  # noqa: E402

MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", str(REPO_ROOT / "pretrained_models" / "CosyVoice-300M"))

DEFAULT_SPEAKER = "en_speaker_6s"
DEFAULT_MAX_SPEECH_TOKENS = 200


def _wav_to_int16(wav: torch.Tensor) -> np.ndarray:
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    wav = wav.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
    return (wav * 32767.0).astype(np.int16)


def _save_wav(path: Path, wav_int16: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(wav_int16.tobytes())


def _build_tt_model(device):
    """Load reference + construct the TT E2E model."""
    print(f"[tts_instruct] Loading reference model from {MODEL_DIR} ...")
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


def _resolve_speaker_inputs(speaker_id: str) -> dict:
    """Look up a registered speaker in ``spk2info.pt`` and return the
    per-speaker inputs needed for ``TtCosyVoiceModel.inference_instruct``.

    The speaker's pre-computed (flow_embedding, prompt_speech_feat, prompt
    speech tokens) are reused so the flow gets a consistent, real voice
    identity. ``llm_embedding`` is dropped internally by the LLM in
    instruct mode, so we don't pass it.
    """
    spk2info = _get_predefined_speakers()
    if speaker_id not in spk2info:
        available = ", ".join(sorted(spk2info.keys()))
        raise ValueError(f"unknown speaker {speaker_id!r}. Available: {available}")
    spk = spk2info[speaker_id]
    return {
        "llm_prompt_speech_token": spk["llm_prompt_speech_token"].clone(),
        "llm_prompt_speech_token_len": spk["llm_prompt_speech_token_len"].clone(),
        "flow_prompt_speech_token": spk["flow_prompt_speech_token"].clone(),
        "flow_prompt_speech_token_len": spk["flow_prompt_speech_token_len"].clone(),
        "prompt_speech_feat": spk["prompt_speech_feat"].clone(),
        "prompt_speech_feat_len": spk["prompt_speech_feat_len"].clone(),
        "flow_embedding": spk["flow_embedding"].clone(),
    }


def run_instruct_tts(
    device,
    text: str,
    instruction: str,
    output_path: Path,
    speaker_id: str = DEFAULT_SPEAKER,
    max_speech_tokens: int = DEFAULT_MAX_SPEECH_TOKENS,
) -> torch.Tensor:
    """Run instruct-mode TTS. Returns the wav tensor and writes it to `output_path`."""
    ref_model, tt_model = _build_tt_model(device)

    torch.manual_seed(0)

    text_token = _encode_text(text)
    text_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)

    instruct_token = _encode_text(instruction)
    instruct_len = torch.tensor([instruct_token.shape[1]], dtype=torch.int32)

    spk = _resolve_speaker_inputs(speaker_id)

    print(
        f"[tts_instruct] text={text!r} ({text_token.shape[1]} tokens), "
        f"instruction={instruction!r} ({instruct_token.shape[1]} tokens), "
        f"speaker={speaker_id!r}, max_speech_tokens={max_speech_tokens}"
    )
    wav = tt_model.inference_instruct(
        text=text_token,
        text_len=text_len,
        instruct_text=instruct_token,
        instruct_text_len=instruct_len,
        llm_prompt_speech_token=spk["llm_prompt_speech_token"],
        llm_prompt_speech_token_len=spk["llm_prompt_speech_token_len"],
        flow_prompt_speech_token=spk["flow_prompt_speech_token"],
        flow_prompt_speech_token_len=spk["flow_prompt_speech_token_len"],
        prompt_speech_feat=spk["prompt_speech_feat"],
        prompt_speech_feat_len=spk["prompt_speech_feat_len"],
        flow_embedding=spk["flow_embedding"],
        max_speech_tokens=max_speech_tokens,
    )
    print(
        f"[tts_instruct] wav shape={tuple(wav.shape)} " f"duration={wav.shape[-1] / TtCosyVoiceModel.SAMPLE_RATE:.2f}s"
    )
    print(
        f"[tts_instruct] wav stats: min={wav.min():.3f} max={wav.max():.3f} "
        f"mean={wav.mean():.3f} std={wav.std():.3f}"
    )

    _save_wav(output_path, _wav_to_int16(wav), TtCosyVoiceModel.SAMPLE_RATE)
    print(f"[tts_instruct] Wrote {output_path}")
    return wav


@pytest.mark.parametrize("mesh_device", (1,), indirect=True)
def test_tts_instruct_runs(mesh_device, tmp_path):
    """Pytest entry: run instruct on the canned English text/instruction pair, save wav to tmp_path."""
    wav = run_instruct_tts(
        device=mesh_device,
        text="Hello world, this is a test of English synthesis.",
        instruction="Speak in a calm and friendly tone.",
        output_path=tmp_path / "instruct_en.wav",
        speaker_id=DEFAULT_SPEAKER,
        max_speech_tokens=DEFAULT_MAX_SPEECH_TOKENS,
    )
    assert wav.dim() in (1, 2)
    assert torch.isfinite(wav).all()
    assert wav.abs().max() <= 0.99 + 1e-3, f"wav out of audio_limit: {wav.abs().max()}"
    assert wav.shape[-1] > 1000, f"wav too short: {wav.shape[-1]} samples"


def main():
    global MODEL_DIR
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument(
        "--text",
        required=True,
        help="TTS text to synthesize. Prepend a language tag (e.g. '<|zh|>') for non-English.",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="Style/tone instruction. Prepend a language tag for non-English.",
    )
    parser.add_argument(
        "--speaker",
        default=DEFAULT_SPEAKER,
        help="Predefined speaker ID (see spk2info.pt). Default: en_speaker_6s.",
    )
    parser.add_argument("--output", default="out.wav")
    parser.add_argument("--max-speech-tokens", type=int, default=DEFAULT_MAX_SPEECH_TOKENS)
    args = parser.parse_args()

    MODEL_DIR = args.model_dir

    device = ttnn.open_device(device_id=0, l1_small_size=64 << 10, trace_region_size=128 << 20)
    device.enable_program_cache()
    try:
        run_instruct_tts(
            device=device,
            text=args.text,
            instruction=args.instruction,
            output_path=Path(args.output),
            speaker_id=args.speaker,
            max_speech_tokens=args.max_speech_tokens,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
