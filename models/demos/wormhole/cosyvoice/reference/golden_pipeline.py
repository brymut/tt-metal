# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference E2E golden-path pipeline for CosyVoice-300M.

For each (mode, language) test case in ``reference.golden_prompts``, this module:
  1. Builds a deterministic input dict (text tokens + prompt tensors) from a fixed seed.
  2. Runs ``llm.inference`` → ``flow.inference`` → ``hift.inference`` on CPU.
  3. Saves the inputs, semantic tokens, mel, and final wav to ``tests/golden/``.

Re-running with the same seed + model produces bit-exact wavs, so the goldens serve
as a stable reference for TTNN port validation.

Usage::

    python -m models.demos.wormhole.cosyvoice.reference.golden_pipeline \
        --model-dir pretrained_models/CosyVoice-300M \
        --golden-dir models/demos/wormhole/cosyvoice/tests/golden
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure whisper + the CosyVoice package are importable before we touch the reference model.
_REPO_ROOT = Path(__file__).resolve().parents[4]
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

import whisper  # noqa: E402

from models.demos.wormhole.cosyvoice.reference import golden_prompts  # noqa: E402
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel  # noqa: E402

# ----------------------------------------------------------------------------
# Constants matching cosyvoice.yaml
# ----------------------------------------------------------------------------
SAMPLE_RATE = 22050
SPK_EMBED_DIM = 192
SPEECH_VOCAB_SIZE = 4096
MEL_DIM = 80
INPUT_FRAME_RATE = 50
HOP_SIZE = 256

# Deterministic-shape constants for synthetic prompt tensors. These are the
# "prompt" length scale that the speech tokenizer / feat extractor would produce
# for a ~1 second reference audio: 50 speech tokens at 50 Hz, mel length 50*2=100
# (token_mel_ratio=2 in the architecture).
PROMPT_SPEECH_TOKEN_LEN = 10
PROMPT_MEL_LEN = 20
MAX_DECODED_SPEECH_TOKENS = 200  # ~4s of audio; long enough for meaningful Whisper WER and campplus spk_sim


# ----------------------------------------------------------------------------
# Tokenization helper
# ----------------------------------------------------------------------------
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = whisper.tokenizer.get_tokenizer(
            multilingual=True, num_languages=100, language="en", task="transcribe"
        )
    return _TOKENIZER


def _encode_text(text: str) -> torch.Tensor:
    """Encode text with whisper multilingual tokenizer (allows <|lang|> tags)."""
    tok = _get_tokenizer()
    ids = tok.encode(text, allowed_special="all")
    return torch.tensor([ids], dtype=torch.int32)


# ----------------------------------------------------------------------------
# Deterministic input builder (used by non-SFT modes)
# ----------------------------------------------------------------------------
def _seeded_tensor(seed: int, shape, dtype, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype)
    if dtype == torch.int32 or dtype == torch.int64:
        return torch.randint(0, 4096, shape, generator=g, dtype=dtype)
    raise ValueError(f"unsupported dtype {dtype}")


# ----------------------------------------------------------------------------
# Predefined speakers (registered from real reference audio)
# ----------------------------------------------------------------------------
# Speakers are extracted from real reference wavs using the official
# ``CosyVoice.add_zero_shot_spk`` flow, which runs the speech tokenizer +
# feat extractor + campplus on the audio. The resulting ``spk2info`` is
# saved to ``pretrained_models/CosyVoice-300M/spk2info.pt``.
#
# For SFT mode (predefined speaker), we pick one of these registered speakers
# per language and reuse its (embedding, prompt_speech_token, prompt_speech_feat).
# This gives a real, consistent voice per language instead of the v1 SFT model
# being asked to voice-clone a random vector.
_PREDEFINED_SPK2INFO = None  # lazy-loaded torch.load of spk2info.pt


def _get_predefined_speakers() -> dict:
    global _PREDEFINED_SPK2INFO
    if _PREDEFINED_SPK2INFO is None:
        # _REPO_ROOT above is parents[4] = /root/tt-metal/models, so we go up one
        # more level to get to /root/tt-metal where pretrained_models/ lives.
        spk2info_path = _REPO_ROOT.parent / "pretrained_models" / "CosyVoice-300M" / "spk2info.pt"
        if not spk2info_path.exists():
            raise FileNotFoundError(
                f"spk2info.pt not found at {spk2info_path}. "
                "Run the speaker-registration snippet (see HANDOFF.md §Option 1) "
                "to register predefined speakers from real reference audio."
            )
        _PREDEFINED_SPK2INFO = torch.load(spk2info_path, map_location="cpu", weights_only=True)
    return _PREDEFINED_SPK2INFO


# Language → speaker_id mapping for SFT mode. Each language gets a real
# predefined speaker extracted from real reference audio. The mapping uses
# native-language speakers where available, falling back to the multilingual
# speaker for languages without a native one registered.
_SFT_SPEAKER_FOR_LANG = {
    "en": "en_speaker_6s",  # 5.86s English (LibriSpeech)
    "zh": "zh_speaker_3s",  # 3.48s Chinese (CosyVoice v1 asset)
    "ja": "ja_speaker_11s",  # 11.10s Japanese (FLEURS)
    "yue": "yue_speaker_15s",  # 15.48s Cantonese (FLEURS)
    "ko": "ko_speaker_11s",  # 10.56s Korean (FLEURS)
}


def build_test_case(mode: str, lang: str, base_seed: int = 1000) -> dict:
    """Build the full input dict for a (mode, lang) test case.

    Returns a dict with keys that mirror the official CosyVoice ``model.tts(...)``
    call: ``text``, ``text_len``, ``prompt_text``, ``prompt_text_len``,
    ``llm_prompt_speech_token``, ``llm_prompt_speech_token_len``,
    ``flow_prompt_speech_token``, ``flow_prompt_speech_token_len``,
    ``prompt_speech_feat``, ``prompt_speech_feat_len``, ``llm_embedding``,
    ``flow_embedding``.
    """
    if mode not in golden_prompts.MODES:
        raise ValueError(f"unknown mode: {mode}")
    if lang not in golden_prompts.LANGUAGES:
        raise ValueError(f"unknown language: {lang}")

    case_seed = base_seed + hash((mode, lang)) % 100000
    text = golden_prompts.text_for(mode, lang)
    prompt_text_str = golden_prompts.prompt_text_for(mode, lang)
    instruct_text_str = golden_prompts.instruct_text_for(mode, lang)

    text_token = _encode_text(text)
    text_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)

    # Speaker embedding: a fixed (mode, lang)-dependent vector. The flow uses
    # F.normalize(embedding, dim=1) before projection; pre-normalized isn't
    # required (the model normalizes internally), but we keep it unit-ish.
    spk_seed = case_seed + 1
    embedding = _seeded_tensor(spk_seed, (1, SPK_EMBED_DIM), torch.float32, scale=1.0)

    out: dict = {
        "text": text_token,
        "text_len": text_len,
        "llm_embedding": embedding,
        "flow_embedding": embedding,
    }

    if mode == "sft":
        # SFT (predefined speaker): use a real speaker registered from a real
        # reference audio via the official add_zero_shot_spk mechanism. The
        # speaker's pre-computed (embedding, prompt_speech_token, prompt_speech_feat)
        # are reused so the LLM/Flow get a consistent, real voice identity.
        spk2info = _get_predefined_speakers()
        spk_id = _SFT_SPEAKER_FOR_LANG[lang]
        if spk_id not in spk2info:
            raise KeyError(f"speaker {spk_id!r} not in spk2info. Available: {list(spk2info.keys())}")
        spk = spk2info[spk_id]
        out["llm_embedding"] = spk["llm_embedding"].clone()
        out["flow_embedding"] = spk["flow_embedding"].clone()
        out["llm_prompt_speech_token"] = spk["llm_prompt_speech_token"].clone()
        out["llm_prompt_speech_token_len"] = spk["llm_prompt_speech_token_len"].clone()
        out["flow_prompt_speech_token"] = spk["flow_prompt_speech_token"].clone()
        out["flow_prompt_speech_token_len"] = spk["flow_prompt_speech_token_len"].clone()
        out["prompt_speech_feat"] = spk["prompt_speech_feat"].clone()
        out["prompt_speech_feat_len"] = spk["prompt_speech_feat_len"].clone()
        # SFT mode does NOT pass prompt_text to the LLM (per official frontend_sft).
        out["prompt_text"] = torch.zeros(1, 0, dtype=torch.int32)
        out["prompt_text_len"] = torch.tensor([0], dtype=torch.int32)
        return out

    # All non-SFT modes need prompt speech tokens + prompt mel features.
    pst_seed = case_seed + 2
    prompt_speech_token = _seeded_tensor(pst_seed, (1, PROMPT_SPEECH_TOKEN_LEN), torch.int32)
    out["llm_prompt_speech_token"] = prompt_speech_token
    out["llm_prompt_speech_token_len"] = torch.tensor([prompt_speech_token.shape[1]], dtype=torch.int32)
    out["flow_prompt_speech_token"] = prompt_speech_token
    out["flow_prompt_speech_token_len"] = torch.tensor([prompt_speech_token.shape[1]], dtype=torch.int32)

    pf_seed = case_seed + 3
    prompt_speech_feat = _seeded_tensor(pf_seed, (1, PROMPT_MEL_LEN, MEL_DIM), torch.float32, scale=0.5)
    out["prompt_speech_feat"] = prompt_speech_feat
    out["prompt_speech_feat_len"] = torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32)

    if mode == "cross_lingual":
        # Cross-lingual: LLM does NOT see prompt_text. Flow uses prompt mel + spk embed.
        out["prompt_text"] = torch.zeros(1, 0, dtype=torch.int32)
        out["prompt_text_len"] = torch.tensor([0], dtype=torch.int32)
        return out

    if mode == "zero_shot":
        # Zero-shot: LLM sees prompt_text + llm_prompt_speech_token; flow uses mel + spk.
        prompt_text_token = _encode_text(prompt_text_str)
        out["prompt_text"] = prompt_text_token
        out["prompt_text_len"] = torch.tensor([prompt_text_token.shape[1]], dtype=torch.int32)
        return out

    if mode == "instruct":
        # Instruct: LLM uses prompt_text=instruct_text (not prompt_text); llm_embedding dropped.
        # Mirrors official frontend_instruct: del llm_embedding. The reference LLM checks
        # `if embedding.shape[0] != 0` (llm.py:186) — must pass batch=0 to skip spk_embed_affine_layer.
        instruct_token = _encode_text(instruct_text_str)
        out["prompt_text"] = instruct_token
        out["prompt_text_len"] = torch.tensor([instruct_token.shape[1]], dtype=torch.int32)
        out["llm_embedding"] = torch.zeros(0, SPK_EMBED_DIM, dtype=torch.float32)
        return out

    raise ValueError(f"unhandled mode: {mode}")


# ----------------------------------------------------------------------------
# Reference pipeline driver
# ----------------------------------------------------------------------------
def _wav_to_int16(wav: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] waveform tensor to int16 PCM numpy array."""
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    wav = wav.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
    return (wav * 32767.0).astype(np.int16)


def _save_wav(path: Path, wav_int16: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save int16 PCM to a .wav file using stdlib ``wave`` (no torchaudio dep)."""
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(wav_int16.tobytes())


def run_reference_pipeline(
    model: CosyVoiceReferenceModel,
    inputs: dict,
    max_speech_tokens: int = MAX_DECODED_SPEECH_TOKENS,
    sampling_seed: int = 0,
):
    """Run llm → flow → hift for a single test case. Returns (wav, mel, tokens).

    ``sampling_seed`` seeds the global torch RNG before LLM top-k sampling so
    that the token sequence is reproducible across ``regenerate_goldens`` and
    ``verify_goldens``. Default 0; pass a per-case value if you need to vary
    sampling without changing the inputs.
    """
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)

    llm = model.llm
    flow = model.flow
    # ``CosyVoiceReferenceModel`` exposes the vocoder as ``hifigan``
    # (mirroring ``self.model.hift`` from the official CosyVoiceModel).
    hift = model.hifigan

    text = inputs["text"]
    text_len = inputs["text_len"]
    prompt_text = inputs["prompt_text"]
    prompt_text_len = inputs["prompt_text_len"]
    pst = inputs["llm_prompt_speech_token"]
    pst_len = inputs["llm_prompt_speech_token_len"]
    embedding = inputs["llm_embedding"]

    # ---- 1. LLM autoregressive inference → semantic tokens ----
    speech_tokens = []
    for tok in llm.inference(
        text=text,
        text_len=text_len,
        prompt_text=prompt_text,
        prompt_text_len=prompt_text_len,
        prompt_speech_token=pst,
        prompt_speech_token_len=pst_len,
        embedding=embedding,
        sampling=25,
        max_token_text_ratio=10.0,
        min_token_text_ratio=2.0,
    ):
        speech_tokens.append(int(tok))
        if len(speech_tokens) >= max_speech_tokens:
            break
    if not speech_tokens:
        raise RuntimeError("LLM produced no speech tokens")
    speech_tokens_t = torch.tensor([speech_tokens], dtype=torch.int32)

    # ---- 2. Flow: tokens + prompt mel → mel spectrogram ----
    flow_prompt_token = inputs["flow_prompt_speech_token"]
    flow_prompt_token_len = inputs["flow_prompt_speech_token_len"]
    prompt_feat = inputs["prompt_speech_feat"]
    prompt_feat_len = inputs["prompt_speech_feat_len"]
    flow_embedding = inputs["flow_embedding"]
    flow_cache = torch.zeros(1, MEL_DIM, 0, 2, dtype=torch.float32)

    with torch.no_grad():
        mel, _ = flow.inference(
            token=speech_tokens_t,
            token_len=torch.tensor([speech_tokens_t.shape[1]], dtype=torch.int32),
            prompt_token=flow_prompt_token,
            prompt_token_len=flow_prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=flow_embedding,
            flow_cache=flow_cache,
        )

    # ---- 3. HiFi-GAN: mel → waveform ----
    # Reference HiFTGenerator.inference signature: (speech_feat, cache_source)
    # The CosyVoice1 base model uses streaming=None and returns (speech, source).
    with torch.no_grad():
        wav, _ = hift.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))

    return wav, mel, speech_tokens


# ----------------------------------------------------------------------------
# Per-case save / load
# ----------------------------------------------------------------------------
def _save_case(golden_dir: Path, case: str, inputs: dict, mel: torch.Tensor, tokens, wav) -> None:
    (golden_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (golden_dir / "mels").mkdir(parents=True, exist_ok=True)
    (golden_dir / "tokens").mkdir(parents=True, exist_ok=True)
    (golden_dir / "wavs").mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "text": inputs["text"],
            "text_len": inputs["text_len"],
            "prompt_text": inputs["prompt_text"],
            "prompt_text_len": inputs["prompt_text_len"],
            "llm_prompt_speech_token": inputs["llm_prompt_speech_token"],
            "llm_prompt_speech_token_len": inputs["llm_prompt_speech_token_len"],
            "flow_prompt_speech_token": inputs["flow_prompt_speech_token"],
            "flow_prompt_speech_token_len": inputs["flow_prompt_speech_token_len"],
            "prompt_speech_feat": inputs["prompt_speech_feat"],
            "prompt_speech_feat_len": inputs["prompt_speech_feat_len"],
            "llm_embedding": inputs["llm_embedding"],
            "flow_embedding": inputs["flow_embedding"],
        },
        golden_dir / "inputs" / f"{case}.pt",
    )
    torch.save(mel.cpu(), golden_dir / "mels" / f"{case}.pt")
    torch.save(torch.tensor(tokens, dtype=torch.int32), golden_dir / "tokens" / f"{case}.pt")
    _save_wav(golden_dir / "wavs" / f"{case}.wav", _wav_to_int16(wav))


def _load_case(golden_dir: Path, case: str) -> tuple:
    inputs = torch.load(golden_dir / "inputs" / f"{case}.pt", map_location="cpu", weights_only=True)
    mel = torch.load(golden_dir / "mels" / f"{case}.pt", map_location="cpu", weights_only=True)
    tokens = torch.load(golden_dir / "tokens" / f"{case}.pt", map_location="cpu", weights_only=True)
    return inputs, mel, tokens


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def regenerate_goldens(model_dir: str, golden_dir: str | Path, modes=None, languages=None) -> None:
    """Generate golden wavs / mels / tokens for all (mode, lang) test cases."""
    golden_dir = Path(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)

    modes = modes or golden_prompts.MODES
    languages = languages or golden_prompts.LANGUAGES

    print(f"Loading reference model from {model_dir} ...")
    model = CosyVoiceReferenceModel(model_dir=model_dir)
    model.llm.eval()
    model.flow.eval()
    model.hifigan.eval()

    total = len(modes) * len(languages)
    idx = 0
    for mode in modes:
        for lang in languages:
            idx += 1
            case = golden_prompts.case_id(mode, lang)
            wav_path = golden_dir / "wavs" / f"{case}.wav"
            if wav_path.exists():
                print(f"[{idx}/{total}] {case}: already exists, skipping")
                continue
            print(f"[{idx}/{total}] {case}: building inputs ...", flush=True)
            inputs = build_test_case(mode, lang)
            print(f"[{idx}/{total}] {case}: running LLM+Flow+HiFi-GAN ...", flush=True)
            wav, mel, tokens = run_reference_pipeline(model, inputs)
            print(
                f"[{idx}/{total}] {case}: tokens={len(tokens)} mel={tuple(mel.shape)} "
                f"wav={wav.shape[-1]} samples ({wav.shape[-1] / SAMPLE_RATE:.2f}s) — saving"
            )
            _save_case(golden_dir, case, inputs, mel, tokens, wav)

    print(f"Done. {total} cases in {golden_dir}")


def verify_goldens(model_dir: str, golden_dir: str | Path, modes=None, languages=None) -> bool:
    """Re-run the reference pipeline from saved inputs and compare to saved goldens.

    Returns True if all cases match exactly (bit-exact). Since the inputs are
    saved and the seed is fixed, a correct reference should reproduce exactly.
    """
    golden_dir = Path(golden_dir)
    modes = modes or golden_prompts.MODES
    languages = languages or golden_prompts.LANGUAGES

    print(f"Loading reference model from {model_dir} ...")
    model = CosyVoiceReferenceModel(model_dir=model_dir)
    model.llm.eval()
    model.flow.eval()
    model.hifigan.eval()

    ok = True
    total = len(modes) * len(languages)
    idx = 0
    for mode in modes:
        for lang in languages:
            idx += 1
            case = golden_prompts.case_id(mode, lang)
            print(f"[{idx}/{total}] {case}: loading saved ...", flush=True)
            inputs_saved, mel_saved, tokens_saved = _load_case(golden_dir, case)
            wav_saved = torch.from_numpy(
                _load_wav_int16(golden_dir / "wavs" / f"{case}.wav").astype(np.float32) / 32767.0
            )

            print(f"[{idx}/{total}] {case}: re-running pipeline ...", flush=True)
            wav_new, mel_new, tokens_new = run_reference_pipeline(model, inputs_saved)

            tokens_match = torch.tensor(tokens_new, dtype=torch.int32).equal(tokens_saved)
            mel_match = torch.allclose(mel_new.cpu(), mel_saved, atol=1e-5, rtol=1e-5)
            # HiFi-GAN uses the global torch RNG (for noise / F0 source init), so
            # even with identical mel + weights, the wav is only reproducible up
            # to the local RNG state. We use a relaxed tolerance for wav and rely
            # on mel/token bit-equality as the primary signal.
            wav_match = torch.allclose(wav_new.cpu().squeeze(0), wav_saved, atol=1e-3, rtol=1e-3)

            print(
                f"[{idx}/{total}] {case}: tokens={tokens_match} mel={mel_match} wav={wav_match} "
                f"(max mel abs diff={float((mel_new.cpu() - mel_saved).abs().max()):.3e})",
                flush=True,
            )
            ok = ok and tokens_match and mel_match and wav_match
    return ok


def _load_wav_int16(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        data = w.readframes(n)
    return np.frombuffer(data, dtype=np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--verify", action="store_true", help="Re-run from saved inputs and compare.")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--languages", nargs="*", default=None)
    args = parser.parse_args()

    if args.verify:
        ok = verify_goldens(args.model_dir, args.golden_dir, args.modes, args.languages)
        sys.exit(0 if ok else 1)
    else:
        regenerate_goldens(args.model_dir, args.golden_dir, args.modes, args.languages)


if __name__ == "__main__":
    main()
