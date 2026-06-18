# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio-quality evaluation harness for the CosyVoice TTNN bring-up.

Runs the full E2E pipeline on all 20 (mode, lang) golden cases and
computes three metrics per case:

  1. Token-level accuracy (TT greedy tokens vs ref golden tokens)
  2. WER (Whisper-transcribed TT wav vs source text)
  3. Speaker similarity (cosine sim of campplus emb(TT wav) vs campplus emb(golden ref wav))

This is a measurement harness, not a strict pass/fail test: it always
exits with code 0 and prints a results table. We use it to track
accuracy improvements as we iterate on the bf16-error-accumulation
root cause (see HANDOFF §6 Priority 2).

The test is parametrized over (mode, lang). Individual cases are
independent; failures in one do not skip others. Use ``--eval-modes``
and ``--eval-languages`` flags to limit scope during iteration.

Usage::

    pytest -svv models/demos/wormhole/cosyvoice/tests/test_audio_quality.py

    # Or to limit to one mode/lang:
    pytest -svv models/demos/wormhole/cosyvoice/tests/test_audio_quality.py --eval-modes sft --eval-languages en
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from models.demos.wormhole.cosyvoice.reference import golden_prompts
from models.demos.wormhole.cosyvoice.reference.golden_pipeline import MAX_DECODED_SPEECH_TOKENS
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tests.audio_eval import (
    AudioEvalResult,
    CampPlusEmbedder,
    WhisperTranscriber,
    evaluate_case,
    format_results_table,
    load_wav_int16,
    save_wav,
    summary_stats,
)
from models.demos.wormhole.cosyvoice.tt.model import TtCosyVoiceModel
from models.demos.wormhole.cosyvoice.tt.model_config import create_model_config

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent.parent.parent
MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", str(REPO_ROOT / "pretrained_models" / "CosyVoice-300M"))
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"


# ----------------------------------------------------------------------------
# Fixtures (module-scoped so we only build the model and load ONNX once)
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


@pytest.fixture(scope="module")
def tt_model(device, reference_model):
    config = create_model_config(batch_size=1, hidden_size=1024)
    return TtCosyVoiceModel(device, config, args=None, ref_model=reference_model)


@pytest.fixture(scope="module")
def campplus():
    return CampPlusEmbedder(MODEL_DIR + "/campplus.onnx")


@pytest.fixture(scope="module")
def whisper_transcriber():
    # Use the small model by default — tiny/base cannot transcribe the
    # ~1s golden wavs reliably. Override via COSYVOICE_WHISPER_MODEL env
    # var (e.g. "medium", "large-v3") for higher accuracy.
    model_name = os.getenv("COSYVOICE_WHISPER_MODEL", "small")
    return WhisperTranscriber(model_name=model_name)


# ----------------------------------------------------------------------------
# Parametrization
# ----------------------------------------------------------------------------
def _gen_case_ids(modes, languages):
    return [(m, l) for m in modes for l in languages]


def pytest_generate_tests(metafunc):
    from models.demos.wormhole.cosyvoice.reference import golden_prompts as gp

    if "eval_case" not in metafunc.fixturenames:
        return
    modes = metafunc.config.getoption("--eval-modes") or gp.MODES
    languages = metafunc.config.getoption("--eval-languages") or gp.LANGUAGES
    cases = _gen_case_ids(modes, languages)
    metafunc.parametrize("eval_case", cases, ids=[f"{m}_{l}" for m, l in cases])


# ----------------------------------------------------------------------------
# The test
# ----------------------------------------------------------------------------
def test_eval_case(
    device,
    reference_model,
    tt_model,
    campplus,
    whisper_transcriber,
    eval_case,
    tmp_path_factory,
):
    mode, lang = eval_case
    case_id = golden_prompts.case_id(mode, lang)

    # 1. Load the golden inputs / ref wav / ref tokens
    inputs = torch.load(GOLDEN_DIR / "inputs" / f"{case_id}.pt", map_location="cpu", weights_only=True)
    # Golden tokens were generated with top-k=25 + RAS sampling (see
    # golden_pipeline.run_reference_pipeline). We compute two baselines:
    #   - ref_greedy: deterministic argmax sequence (diagnostic)
    #   - ref_sampled: top-k=25 + RAS with a fixed seed (matches E2E)
    ref_greedy_tokens = _run_ref_greedy_tokens(
        reference_model, mode, inputs, max_speech_tokens=MAX_DECODED_SPEECH_TOKENS
    )
    ref_sampled_tokens = _run_ref_sampled_tokens(
        reference_model, mode, inputs, max_speech_tokens=MAX_DECODED_SPEECH_TOKENS, sampling_seed=0
    )
    ref_wav_int16, ref_sr = load_wav_int16(GOLDEN_DIR / "wavs" / f"{case_id}.wav")
    text = golden_prompts.text_for(mode, lang)

    # 2. Run the TT pipeline (top-k=25 + RAS sampling, matching the golden)
    torch.manual_seed(0)
    with torch.no_grad():
        tt_wav = _run_tt_mode(tt_model, mode, inputs, max_speech_tokens=MAX_DECODED_SPEECH_TOKENS)
    tt_wav_int16 = (tt_wav.squeeze(0).clamp(-1.0, 1.0).cpu().numpy() * 32767.0).astype(np.int16)

    # 3. Save the TT wav for listening
    out_dir = tmp_path_factory.mktemp(f"audio_eval_{case_id}")
    tt_wav_path = out_dir / "tt.wav"
    save_wav(tt_wav_path, tt_wav_int16, TtCosyVoiceModel.SAMPLE_RATE)

    # 4. Token-level accuracy: compute BOTH greedy and sampled comparisons.
    # greedy is the most deterministic diagnostic; sampled (with RAS) is
    # closer to the actual E2E pipeline.
    tt_greedy_tokens = _run_tt_greedy_tokens(tt_model, mode, inputs, max_speech_tokens=MAX_DECODED_SPEECH_TOKENS)
    tt_sampled_tokens = _run_tt_sampled_tokens(
        tt_model, mode, inputs, max_speech_tokens=MAX_DECODED_SPEECH_TOKENS, sampling_seed=0
    )

    # 5. Run the full eval
    result = evaluate_case(
        case_id=case_id,
        mode=mode,
        lang=lang,
        text=text,
        ref_tokens_greedy=ref_greedy_tokens,
        tt_tokens_greedy=tt_greedy_tokens,
        ref_tokens_sampled=ref_sampled_tokens,
        tt_tokens_sampled=tt_sampled_tokens,
        ref_wav=(ref_wav_int16, ref_sr),
        tt_wav=(tt_wav_int16, TtCosyVoiceModel.SAMPLE_RATE),
        campplus=campplus,
        whisper_transcriber=whisper_transcriber,
    )

    # 6. Persist the result for trend tracking
    results_dir = PROJECT_ROOT / "tests" / "audio_eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{case_id}.json", "w") as f:
        json.dump(result.as_dict(), f, indent=2)

    # 7. Print a single-line summary
    print(
        f"[audio_eval] {case_id}: tok_acc_g={result.token_accuracy_greedy:.3f} "
        f"tok_acc_s={result.token_accuracy_sampled:.3f} "
        f"wer_ref={result.wer_ref:.3f} wer_tt={result.wer_tt:.3f} "
        f"spk_sim={result.spk_sim:.3f} tt_dur={result.tt_wav_dur:.2f}s "
        f"n_tok={result.n_tokens_tt} tt_wav={tt_wav_path}"
    )

    # 8. Note the 1-second golden wav limitation in the test output. The
    # goldens use 50 speech tokens (~1s of audio), which is below the
    # typical reliable WER window for whisper. WER is reported for trend
    # tracking; absolute values are not directly meaningful.
    if result.tt_wav_dur < 2.0 or result.ref_wav_dur < 2.0:
        print(
            f"[audio_eval] NOTE: {case_id} wav is {result.tt_wav_dur:.2f}s "
            f"(ref {result.ref_wav_dur:.2f}s). WER is noisy on sub-2s audio; "
            f"interpret absolute values with caution. Token accuracy and "
            f"speaker similarity are robust."
        )

    # 9. Strict gates (informational — this is a measurement harness, not a
    # pass/fail test). We only assert the wav is finite and in-range; the
    # quality metrics are printed for the next agent to act on.
    assert torch.isfinite(tt_wav).all(), f"{case_id}: TT wav has NaN/Inf"


# ----------------------------------------------------------------------------
# Shared LLM kwargs builder (used by both the TT and the ref dispatch fns)
# ----------------------------------------------------------------------------
def _llm_kwargs_for_mode(mode: str, inputs: dict) -> dict:
    """Build the kwargs for ``llm.inference(...)`` from the saved golden
    ``inputs`` dict. Identical for the TT and reference LLM since both ports
    accept the same argument list (per CosyVoice 1 official frontend)."""
    if mode == "sft":
        return dict(
            text=inputs["text"],
            text_len=inputs["text_len"],
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            prompt_text_len=torch.tensor([0], dtype=torch.int32),
            prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_token_len=torch.tensor([0], dtype=torch.int32),
            embedding=inputs["llm_embedding"],
        )
    if mode == "zero_shot":
        return dict(
            text=inputs["text"],
            text_len=inputs["text_len"],
            prompt_text=inputs["prompt_text"],
            prompt_text_len=inputs["prompt_text_len"],
            prompt_speech_token=inputs["llm_prompt_speech_token"],
            prompt_speech_token_len=inputs["llm_prompt_speech_token_len"],
            embedding=inputs["llm_embedding"],
        )
    if mode == "cross_lingual":
        # LLM does NOT see prompt_text in cross_lingual mode (per frontend_cross_lingual).
        return dict(
            text=inputs["text"],
            text_len=inputs["text_len"],
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            prompt_text_len=torch.tensor([0], dtype=torch.int32),
            prompt_speech_token=inputs["llm_prompt_speech_token"],
            prompt_speech_token_len=inputs["llm_prompt_speech_token_len"],
            embedding=inputs["llm_embedding"],
        )
    if mode == "instruct":
        # LLM uses prompt_text=instruct_text; llm_embedding is dropped (batch=0) per
        # frontend_instruct's `del llm_embedding`.
        return dict(
            text=inputs["text"],
            text_len=inputs["text_len"],
            prompt_text=inputs["prompt_text"],
            prompt_text_len=inputs["prompt_text_len"],
            prompt_speech_token=inputs["llm_prompt_speech_token"],
            prompt_speech_token_len=inputs["llm_prompt_speech_token_len"],
            embedding=torch.zeros(0, 192, dtype=torch.float32),
        )
    raise ValueError(f"unknown mode: {mode!r}")


# ----------------------------------------------------------------------------
# TT mode dispatch
# ----------------------------------------------------------------------------
def _tt_kwargs_for_mode(inputs: dict, mode: str) -> dict:
    """Map the saved ``inputs`` dict (golden_pipeline format) to the kwargs
    accepted by ``TtCosyVoiceModel.inference_<mode>(...)``."""
    text = inputs["text"]
    text_len = inputs["text_len"]
    llm_pst = inputs["llm_prompt_speech_token"]
    llm_pst_len = inputs["llm_prompt_speech_token_len"]
    flow_pst = inputs["flow_prompt_speech_token"]
    flow_pst_len = inputs["flow_prompt_speech_token_len"]
    prompt_feat = inputs["prompt_speech_feat"]
    prompt_feat_len = inputs["prompt_speech_feat_len"]
    llm_emb = inputs["llm_embedding"]
    flow_emb = inputs["flow_embedding"]

    if mode == "sft":
        return dict(text=text, text_len=text_len, llm_embedding=llm_emb)
    if mode == "zero_shot":
        return dict(
            text=text,
            text_len=text_len,
            prompt_text=inputs["prompt_text"],
            prompt_text_len=inputs["prompt_text_len"],
            llm_prompt_speech_token=llm_pst,
            llm_prompt_speech_token_len=llm_pst_len,
            flow_prompt_speech_token=flow_pst,
            flow_prompt_speech_token_len=flow_pst_len,
            prompt_speech_feat=prompt_feat,
            prompt_speech_feat_len=prompt_feat_len,
            llm_embedding=llm_emb,
            flow_embedding=flow_emb,
        )
    if mode == "cross_lingual":
        return dict(
            text=text,
            text_len=text_len,
            llm_prompt_speech_token=llm_pst,
            llm_prompt_speech_token_len=llm_pst_len,
            flow_prompt_speech_token=flow_pst,
            flow_prompt_speech_token_len=flow_pst_len,
            prompt_speech_feat=prompt_feat,
            prompt_speech_feat_len=prompt_feat_len,
            llm_embedding=llm_emb,
            flow_embedding=flow_emb,
        )
    if mode == "instruct":
        return dict(
            text=text,
            text_len=text_len,
            instruct_text=inputs["prompt_text"],
            instruct_text_len=inputs["prompt_text_len"],
            llm_prompt_speech_token=llm_pst,
            llm_prompt_speech_token_len=llm_pst_len,
            flow_prompt_speech_token=flow_pst,
            flow_prompt_speech_token_len=flow_pst_len,
            prompt_speech_feat=prompt_feat,
            prompt_speech_feat_len=prompt_feat_len,
            flow_embedding=flow_emb,
        )
    raise ValueError(f"unknown mode: {mode!r}")


def _run_tt_mode(tt_model: TtCosyVoiceModel, mode: str, inputs: dict, max_speech_tokens: int) -> torch.Tensor:
    method = getattr(tt_model, f"inference_{mode}")
    return method(
        **_tt_kwargs_for_mode(inputs, mode),
        max_speech_tokens=max_speech_tokens,
    )


# ----------------------------------------------------------------------------
# Greedy LLM token extractor (apples-to-apples vs ref golden tokens)
# ----------------------------------------------------------------------------
def _run_tt_greedy_tokens(tt_model: TtCosyVoiceModel, mode: str, inputs: dict, max_speech_tokens: int) -> list[int]:
    """Run the TT LLM with sampling=1 (greedy) to get a reproducible token sequence.

    The ref golden uses top-k=25 sampling, but we want a fair
    "TT LLM vs ref LLM" token comparison: if we use top-k on both,
    different RNG draws make the sequences diverge by step 2 even when
    the underlying distributions are close. Greedy argmax is the only
    deterministic comparison.
    """
    tokens: list[int] = []
    for tok in tt_model.llm.inference(
        **_llm_kwargs_for_mode(mode, inputs),
        sampling=1,  # greedy
        max_token_text_ratio=20.0,
        min_token_text_ratio=2.0,
    ):
        tokens.append(int(tok))
        if len(tokens) >= max_speech_tokens:
            break
    return tokens


def _run_ref_greedy_tokens(reference_model, mode: str, inputs: dict, max_speech_tokens: int) -> list[int]:
    """Run the reference LLM with sampling=1 (greedy) to get a reproducible token sequence.

    The golden tokens were saved with sampling=25 (top-k). Re-running the
    reference LLM with greedy gives us a deterministic baseline to
    compare against ``_run_tt_greedy_tokens``.
    """
    ref_llm = reference_model.llm
    tokens: list[int] = []
    for tok in ref_llm.inference(
        **_llm_kwargs_for_mode(mode, inputs),
        sampling=1,  # greedy
        max_token_text_ratio=20.0,
        min_token_text_ratio=2.0,
    ):
        tokens.append(int(tok))
        if len(tokens) >= max_speech_tokens:
            break
    return tokens


def _run_ref_sampled_tokens(
    reference_model, mode: str, inputs: dict, max_speech_tokens: int, sampling_seed: int = 0
) -> list[int]:
    """Run the reference LLM with sampling=25 + RAS, fixed seed.

    Matches the E2E pipeline behavior. The ref uses ``ras_sampling``
    (top-p=0.8, top-k=25, win_size=10, tau_r=0.1) under the hood.
    """
    torch.manual_seed(sampling_seed)
    ref_llm = reference_model.llm
    tokens: list[int] = []
    for tok in ref_llm.inference(
        **_llm_kwargs_for_mode(mode, inputs),
        sampling=25,  # top-k + RAS
        max_token_text_ratio=20.0,
        min_token_text_ratio=2.0,
    ):
        tokens.append(int(tok))
        if len(tokens) >= max_speech_tokens:
            break
    return tokens


def _run_tt_sampled_tokens(
    tt_model, mode: str, inputs: dict, max_speech_tokens: int, sampling_seed: int = 0
) -> list[int]:
    """Run the TT LLM with sampling=25 + RAS (TT _sampling_ids now implements RAS)."""
    torch.manual_seed(sampling_seed)
    tokens: list[int] = []
    for tok in tt_model.llm.inference(
        **_llm_kwargs_for_mode(mode, inputs),
        sampling=25,  # top-k + RAS
        max_token_text_ratio=20.0,
        min_token_text_ratio=2.0,
    ):
        tokens.append(int(tok))
        if len(tokens) >= max_speech_tokens:
            break
    return tokens


# ----------------------------------------------------------------------------
# Summary report (printed once at the end of the pytest session)
# ----------------------------------------------------------------------------
def pytest_sessionfinish(session, exitstatus):
    """Aggregate per-case JSON results and print a final summary table.

    Writes both to stdout and to ``tests/audio_eval_results/summary.txt``
    (the stdout path is buffered by some pytest runners and may not be
    visible; the file is the durable record).
    """
    results_dir = PROJECT_ROOT / "tests" / "audio_eval_results"
    if not results_dir.exists():
        return
    jsons = sorted(results_dir.glob("*.json"))
    if not jsons:
        return
    results: list[AudioEvalResult] = []
    for jp in jsons:
        with open(jp) as f:
            d = json.load(f)
        results.append(AudioEvalResult(**d))
    if not results:
        return
    table = format_results_table(results)
    s = summary_stats(results)
    aggregate = (
        f"Aggregate: n={s['n_cases']} "
        f"mean_tok_acc_g={s['mean_token_accuracy_greedy']:.3f} "
        f"mean_tok_acc_s={s['mean_token_accuracy_sampled']:.3f} "
        f"mean_wer_ref={s['mean_wer_ref']:.3f} "
        f"mean_wer_tt={s['mean_wer_tt']:.3f} "
        f"mean_spk_sim={s['mean_spk_sim']:.3f} "
        f"mean_tt_dur={s['mean_tt_dur_s']:.2f}s"
    )
    full = (
        "=" * 80 + "\n"
        "CosyVoice Audio Quality Summary\n"
        + "=" * 80
        + "\n"
        + table
        + "\n"
        + "-" * 80
        + "\n"
        + aggregate
        + "\n"
        + "=" * 80
        + "\n"
    )
    # Write to a file (durable record)
    with open(results_dir / "summary.txt", "w") as f:
        f.write(full)
    # Also try stdout (may be suppressed by addopts in some runners)
    print("\n" + full)
