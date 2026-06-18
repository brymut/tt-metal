# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Register predefined SFT speakers from real reference audio and save spk2info.pt.

Run this once after cloning the repo (or whenever you want to add/change speakers).
After registration, the saved ``spk2info.pt`` is reused by ``golden_pipeline.py``
when generating SFT-mode goldens, and by anyone running the model on another
machine — no audio files needed at inference time.

Reference audio provenance (all publicly licensed):

  - zh_speaker_3s     CosyVoice v1 official asset, 3.48s Chinese
                      (https://github.com/FunAudioLLM/CosyVoice, Apache-2.0)
  - xling_speaker_14s CosyVoice v1 official asset, 13.75s multi-lingual
                      (https://github.com/FunAudioLLM/CosyVoice, Apache-2.0)
  - en_ref            LibriSpeech dev-clean (hf-internal-testing dummy), 5.86s English
                      (https://www.openslr.org/12, CC-BY-4.0)
  - ja_ref            FLEURS ja_jp validation split, 11.10s Japanese
                      (https://huggingface.co/datasets/google/fleurs, CC-BY-4.0)
  - ko_ref            FLEURS ko_kr validation split, 10.56s Korean
                      (https://huggingface.co/datasets/google/fleurs, CC-BY-4.0)
  - yue_ref           FLEURS yue_hant_hk validation split, 15.48s Cantonese
                      (https://huggingface.co/datasets/google/fleurs, CC-BY-4.0)

Usage::

    source python_env/bin/activate
    python -m models.demos.wormhole.cosyvoice.reference.register_predefined_speakers
"""

import sys
from pathlib import Path

import torch

# Make sure the CosyVoice package is importable. This file is at
# models/demos/wormhole/cosyvoice/reference/register_predefined_speakers.py,
# so parents[5] is the actual repo root /root/tt-metal.
_REPO_ROOT = Path(__file__).resolve().parents[5]
for p in [
    _REPO_ROOT / "models" / "demos" / "wormhole" / "cosyvoice" / "reference",
    _REPO_ROOT / "models" / "demos" / "wormhole" / "cosyvoice" / "reference" / "CosyVoice",
    _REPO_ROOT
    / "models"
    / "demos"
    / "wormhole"
    / "cosyvoice"
    / "reference"
    / "CosyVoice"
    / "third_party"
    / "Matcha-TTS",
]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

MODEL_DIR = _REPO_ROOT / "pretrained_models" / "CosyVoice-300M"
ASSET_DIR = MODEL_DIR / "asset"
SFT_REFS_DIR = ASSET_DIR / "sft_refs"
SPK2INFO_OUT = MODEL_DIR / "spk2info.pt"
# (speaker_id, audio_path, prompt_text) — transcripts match the audio content.
# prompt_text is only used for the LLM's prompt conditioning (not used in SFT mode
# since SFT omits prompt_text per official frontend_sft), so the exact transcript
# isn't critical for SFT goldens — but it's used by zero_shot and matters there.
SPEAKERS = [
    # From CosyVoice v1 official assets
    (
        "zh_speaker_3s",
        ASSET_DIR / "zero_shot_prompt.wav",
        "希望你以后能够做的比我还好呦。",
    ),
    (
        "xling_speaker_14s",
        ASSET_DIR / "cross_lingual_prompt.wav",
        "I am a multilingual speaker and I can speak many languages fluently and naturally with proper pronunciation.",
    ),
    # Downloaded from public datasets
    (
        "en_speaker_6s",
        SFT_REFS_DIR / "en_ref.wav",
        "Mister Quilter is the apostle of the middle classes and we are glad to welcome his gospel.",
    ),
    (
        "ja_speaker_11s",
        SFT_REFS_DIR / "ja_ref.wav",
        "多くの場合 海外のギャップイヤーコースに入学することで 帰国後に実際に大学に進学しやすくなります",
    ),
    (
        "ko_speaker_11s",
        SFT_REFS_DIR / "ko_ref.wav",
        "이제 과학적 데이터가 거대한 규모의 이 탄소 경제는 지난 이백만 년 동안 인류의 진화를 도왔던 안정적인 상태에서 생물권을 몰아냈음을 보여줍니다",
    ),
    (
        "yue_speaker_15s",
        SFT_REFS_DIR / "yue_ref.wav",
        "傷 員 送 往 醫 院 後 打 鬥 停 止 這 時 剩 下 的 囚 犯 中 約 有 40 人 留 在 庭 院 中 拒 絕 回 去 自 己 的 牢 房",
    ),
]


def register_speakers(model_dir: Path = MODEL_DIR, output: Path = SPK2INFO_OUT) -> None:
    """Load the v1 CosyVoice model, register all predefined speakers, save spk2info.pt."""
    print(f"Loading model from {model_dir} ...")
    cv = AutoModel(model_dir=str(model_dir))
    print(f"Model class: {type(cv.model).__name__} (v1 / v2 / v3)")
    print(f"Sample rate: {cv.sample_rate}")

    for spk_id, audio_path, prompt_text in SPEAKERS:
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio for speaker {spk_id!r} not found at {audio_path}. "
                "See module docstring for download instructions."
            )
        print(f"  Registering {spk_id} from {audio_path.name} ...")
        cv.add_zero_shot_spk(prompt_text, str(audio_path), spk_id)

    print(f"\nRegistered {len(cv.frontend.spk2info)} speakers: {list(cv.frontend.spk2info.keys())}")

    # Save the spk2info dict (all the pre-computed per-speaker tensors)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cv.frontend.spk2info, output)
    print(f"Saved spk2info to {output} ({output.stat().st_size / 1024:.1f} KB)")

    # Print per-speaker tensor shapes
    for spk_id, info in cv.frontend.spk2info.items():
        print(f"\n  {spk_id}:")
        for k, v in info.items():
            if torch.is_tensor(v):
                print(f"    {k:32s} shape={tuple(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    register_speakers()
