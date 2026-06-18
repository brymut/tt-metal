# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio-quality evaluation utilities for the CosyVoice TTNN bring-up.

Three metrics, computed per (mode, lang) test case:
  1. Token accuracy: fraction of greedy LLM speech tokens that match the
     golden reference token sequence (position-by-position exact match).
     Note: the LLM uses top-k sampling, so even ref-vs-ref would diverge
     after the first token. For an apples-to-apples comparison, we compare
     greedy argmax tokens (no sampling) against the golden token sequence.
  2. WER (Word Error Rate): Whisper-transcribe the generated wav, compare
     to the golden source text. Computed via the standard
     ``jiwer.wer`` metric. We also report WER on the golden ref wav
     itself as a sanity floor.
  3. Speaker similarity: cosine similarity of the campplus speaker
     embedding extracted from the generated wav and the golden ref wav.
     Both wavs go through the same kaldi-fbank + per-utterance mean-norm
     pipeline as the official CosyVoice ``EmbeddingExtractor``.

All three are written to be tolerant of the current accuracy gap (TT
model diverges from the reference after the first LLM step) so that this
module can serve as a tracking harness, not a strict pass/fail gate.
"""

from __future__ import annotations

import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


# ----------------------------------------------------------------------------
# Audio I/O
# ----------------------------------------------------------------------------
def load_wav_int16(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM mono wav file. Returns (samples, sample_rate)."""
    path = Path(path)
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16)
    return data, sr


def wav_tensor_to_int16(wav: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] waveform tensor to int16 PCM numpy array."""
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    wav = wav.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
    return (wav * 32767.0).astype(np.int16)


def save_wav(path: str | Path, wav_int16: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setnframes(len(wav_int16))
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(wav_int16.tobytes())


def resample_to_16k(wav_int16: np.ndarray, orig_sr: int) -> torch.Tensor:
    """Resample a 16-bit PCM int16 numpy array to 16 kHz. Returns [1, T] float32."""
    wav = torch.from_numpy(wav_int16.astype(np.float32) / 32767.0).unsqueeze(0)
    if orig_sr == 16000:
        return wav
    return torchaudio.functional.resample(wav, orig_sr, 16000)


# ----------------------------------------------------------------------------
# Speaker embedding (campplus.onnx)
# ----------------------------------------------------------------------------
class CampPlusEmbedder:
    """Wrap campplus.onnx with the kaldi-fbank preprocessing used by the
    official CosyVoice ``EmbeddingExtractor`` (utils/onnx.py:36-47).
    """

    def __init__(self, onnx_path: str | Path, max_len_seconds: float = 10.0):
        self.onnx_path = str(onnx_path)
        self.max_len_samples = int(max_len_seconds * 16000)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, wav_int16: np.ndarray, sample_rate: int) -> np.ndarray:
        """Returns a 192-dim float32 speaker embedding (L2-normalized via fbank mean-norm)."""
        wav_16k = resample_to_16k(wav_int16, sample_rate)
        if wav_16k.shape[1] > self.max_len_samples:
            # Center-crop deterministically (no random start, to keep eval reproducible)
            start = (wav_16k.shape[1] - self.max_len_samples) // 2
            wav_16k = wav_16k[:, start : start + self.max_len_samples]
        feat = kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.sess.run(None, {self.input_name: feat.unsqueeze(0).numpy()})[0]
        return emb.flatten().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ----------------------------------------------------------------------------
# Token accuracy
# ----------------------------------------------------------------------------
def token_accuracy(tt_tokens: list[int], ref_tokens: list[int]) -> tuple[float, int, int]:
    """Position-by-position exact-match fraction.

    Returns (accuracy, num_match, num_compared) where num_compared is the
    length of the shorter sequence (positions where the model has already
    emitted a token).
    """
    n = min(len(tt_tokens), len(ref_tokens))
    if n == 0:
        return 0.0, 0, 0
    match = sum(1 for i in range(n) if tt_tokens[i] == ref_tokens[i])
    return match / n, match, n


# ----------------------------------------------------------------------------
# WER via Whisper
# ----------------------------------------------------------------------------
class WhisperTranscriber:
    """Lazy-loaded Whisper transcriber. Uses the ``tiny`` model by default to
    keep CI/runtime low; pass ``model_name="base"`` or larger for higher
    accuracy on the 5 languages.
    """

    def __init__(self, model_name: str = "tiny", device: str = "cpu"):
        import whisper

        self.model_name = model_name
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        # Per-language: Whisper needs the language hint to avoid picking a
        # wrong one for short non-English clips. The whisper tokenizer
        # includes a Language enum we can use to set the option.
        self._tok = whisper.tokenizer.get_tokenizer(
            multilingual=True, num_languages=100, language="en", task="transcribe"
        )

    def transcribe(self, wav_int16: np.ndarray, sample_rate: int, language: str | None = None) -> str:
        """Transcribe a 16-bit PCM int16 wav. Returns lowercased text."""
        import whisper

        wav_16k = resample_to_16k(wav_int16, sample_rate).squeeze(0).numpy()
        # whisper expects float32 in [-1, 1] at 16 kHz
        result = whisper.transcribe(self.model, wav_16k, language=language, fp16=False, task="transcribe")
        return (result.get("text") or "").strip().lower()


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute WER (Word Error Rate) using jiwer. Returns 0.0 for empty ref.

    Strips Whisper special-token tags (``<|en|>``, ``<|zh|>``, ``<|jp|>``,
    ``<|yue|>``, ``<|ko|>``) from both inputs before comparison, so the
    reference text matches what whisper actually transcribes.
    """
    from jiwer import wer

    hypothesis = _strip_whisper_tags(hypothesis)
    reference = _strip_whisper_tags(reference)
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    try:
        return float(wer(reference.strip().lower(), hypothesis.strip().lower()))
    except Exception:
        return 1.0


def _strip_whisper_tags(s: str) -> str:
    """Remove Whisper language/task tags (e.g. ``<|zh|>``, ``<|transcribe|>``)
    and any leading whitespace. Tags in the official CosyVoice texts use
    ``zh``/``jp``/``yue``/``ko``; whisper output uses ISO codes like ``en``.
    The regex covers all of them.
    """
    import re

    return re.sub(r"<\|[^|]*\|>", "", s).strip()


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class AudioEvalResult:
    case_id: str
    mode: str
    lang: str
    text: str
    n_tokens_ref: int
    n_tokens_tt: int
    token_accuracy_greedy: float  # TT greedy vs ref greedy (diagnostic of bf16 error)
    n_token_matches_greedy: int
    token_accuracy_sampled: float  # TT top-k=25+RAS vs ref top-k=25+RAS, fixed seed (closer to E2E)
    n_token_matches_sampled: int
    wer_ref: float
    wer_tt: float
    spk_sim: float
    ref_wav_dur: float
    tt_wav_dur: float

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Per-case eval driver
# ----------------------------------------------------------------------------
def evaluate_case(
    case_id: str,
    mode: str,
    lang: str,
    text: str,
    ref_tokens_greedy: list[int],
    tt_tokens_greedy: list[int],
    ref_tokens_sampled: list[int],
    tt_tokens_sampled: list[int],
    ref_wav: tuple[np.ndarray, int],
    tt_wav: tuple[np.ndarray, int],
    campplus: CampPlusEmbedder,
    whisper_transcriber: WhisperTranscriber | None = None,
) -> AudioEvalResult:
    """Compute all metrics for one (mode, lang) case.

    Two token accuracy metrics are reported:
      * ``greedy``: argmax-argmax (sampling=1). The most deterministic,
        most directly comparable signal. A high value here means the
        TT LLM logits match the ref LLM logits exactly. Will be ~0% if
        bf16 error accumulation in the encoder perturbs argmax.
      * ``sampled``: top-k=25 + RAS, with a fixed seed. Closer to actual
        E2E behavior; uses diversity and similarity rather than exact
        token match.
    """
    ref_int16, ref_sr = ref_wav
    tt_int16, tt_sr = tt_wav

    acc_g, n_match_g, _ = token_accuracy(tt_tokens_greedy, ref_tokens_greedy)
    acc_s, n_match_s, _ = token_accuracy(tt_tokens_sampled, ref_tokens_sampled)

    spk_sim = cosine_similarity(
        campplus(ref_int16, ref_sr),
        campplus(tt_int16, tt_sr),
    )

    wer_ref = wer_tt = float("nan")
    if whisper_transcriber is not None:
        try:
            hyp_ref = whisper_transcriber.transcribe(ref_int16, ref_sr, language=lang_to_whisper_code(lang))
            hyp_tt = whisper_transcriber.transcribe(tt_int16, tt_sr, language=lang_to_whisper_code(lang))
            wer_ref = compute_wer(hyp_ref, text)
            wer_tt = compute_wer(hyp_tt, text)
        except Exception as e:
            print(f"[audio_eval] {case_id}: whisper failed: {e}")

    return AudioEvalResult(
        case_id=case_id,
        mode=mode,
        lang=lang,
        text=text,
        n_tokens_ref=len(ref_tokens_sampled),
        n_tokens_tt=len(tt_tokens_sampled),
        token_accuracy_greedy=acc_g,
        n_token_matches_greedy=n_match_g,
        token_accuracy_sampled=acc_s,
        n_token_matches_sampled=n_match_s,
        wer_ref=wer_ref,
        wer_tt=wer_tt,
        spk_sim=spk_sim,
        ref_wav_dur=len(ref_int16) / ref_sr,
        tt_wav_dur=len(tt_int16) / tt_sr,
    )


def lang_to_whisper_code(lang: str) -> str:
    """Map CosyVoice language code to Whisper language code (2-letter ISO 639-1)."""
    return {
        "en": "en",
        "zh": "zh",
        "ja": "ja",
        "ko": "ko",
        "yue": "zh",  # Whisper doesn't have a Cantonese code; use zh (best-effort)
    }.get(lang, lang)


# ----------------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------------
def format_results_table(results: list[AudioEvalResult]) -> str:
    """Format eval results as an aligned text table."""
    if not results:
        return "(no results)"
    headers = [
        "case",
        "mode",
        "lang",
        "n_tok_ref/tt",
        "tok_acc_g",
        "tok_acc_s",
        "wer_ref",
        "wer_tt",
        "spk_sim",
        "tt_dur(s)",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.case_id,
                r.mode,
                r.lang,
                f"{r.n_tokens_ref}/{r.n_tokens_tt}",
                f"{r.token_accuracy_greedy:.3f}",
                f"{r.token_accuracy_sampled:.3f}",
                f"{r.wer_ref:.3f}" if not np.isnan(r.wer_ref) else "-",
                f"{r.wer_tt:.3f}" if not np.isnan(r.wer_tt) else "-",
                f"{r.spk_sim:.3f}",
                f"{r.tt_wav_dur:.2f}",
            ]
        )
    # column widths
    col_w = [max(len(str(x)) for x in [h] + [row[i] for row in rows]) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    sep = "  ".join("-" * w for w in col_w)
    body = "\n".join("  ".join(str(c).ljust(w) for c, w in zip(row, col_w)) for row in rows)
    return f"{line}\n{sep}\n{body}"


def summary_stats(results: list[AudioEvalResult]) -> dict:
    """Aggregate mean stats over all cases."""
    import statistics

    def safe_mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
        return statistics.mean(xs) if xs else float("nan")

    return {
        "n_cases": len(results),
        "mean_token_accuracy_greedy": safe_mean([r.token_accuracy_greedy for r in results]),
        "mean_token_accuracy_sampled": safe_mean([r.token_accuracy_sampled for r in results]),
        "mean_wer_ref": safe_mean([r.wer_ref for r in results]),
        "mean_wer_tt": safe_mean([r.wer_tt for r in results]),
        "mean_spk_sim": safe_mean([r.spk_sim for r in results]),
        "mean_tt_dur_s": safe_mean([r.tt_wav_dur for r in results]),
    }
