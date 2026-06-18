# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_llm import TtCosyVoiceLLM
from models.demos.wormhole.cosyvoice.tt.model_config import create_model_config

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = os.getenv(
    "COSYVOICE_MODEL_DIR",
    str(PROJECT_ROOT.parent.parent.parent.parent) + "/pretrained_models/CosyVoice-300M",
)


def _greedy(logits):
    return int(torch.argmax(logits, dim=-1).item())


@torch.inference_mode()
def _ref_first_token_logits(ref_llm, text, text_len, embedding):
    """Reference first-token logit vector using the same `forward_chunk` path the official `inference()` uses."""
    text_emb = ref_llm.text_embedding(text)
    encoder_out, _ = ref_llm.text_encoder(text_emb, text_len, decoding_chunk_size=1, num_decoding_left_chunks=-1)
    encoder_out = ref_llm.text_encoder_affine_layer(encoder_out)

    emb = torch.nn.functional.normalize(embedding, dim=1)
    emb = ref_llm.spk_embed_affine_layer(emb).unsqueeze(1)

    sos_emb = ref_llm.llm_embedding.weight[ref_llm.sos].reshape(1, 1, -1)
    task_id_emb = ref_llm.llm_embedding.weight[ref_llm.task_id].reshape(1, 1, -1)
    lm_input = torch.cat([sos_emb, emb, encoder_out, task_id_emb], dim=1)

    out, _, _ = ref_llm.llm.forward_chunk(
        lm_input,
        offset=0,
        required_cache_size=-1,
        att_cache=torch.zeros(0, 0, 0, 0),
        cnn_cache=torch.zeros(0, 0, 0, 0),
        att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), dtype=torch.bool)),
    )
    return ref_llm.llm_decoder(out[:, -1])


@torch.inference_mode()
def _tt_first_token_logits(tt_llm, text, text_len, embedding):
    """TT first-token logit vector by replaying the prefix build + first `forward_chunk` call from `inference()`."""
    prompt_text = torch.zeros(1, 0, dtype=torch.int32)
    prompt_speech = torch.zeros(1, 0, dtype=torch.int32)
    text_full = torch.cat([prompt_text, text], dim=1) if prompt_text.shape[1] > 0 else text
    text_encoded = tt_llm._encode_text(text_full)

    if embedding.shape[0] != 0:
        emb = torch.nn.functional.normalize(embedding, dim=1)
        emb = tt_llm.spk_embed_affine_layer(emb).unsqueeze(1)
    else:
        emb = torch.zeros(1, 0, 1024, dtype=text_encoded.dtype)

    sos_emb = tt_llm.llm_embedding.weight[tt_llm.sos].reshape(1, 1, -1)
    task_id_emb = tt_llm.llm_embedding.weight[tt_llm.task_id].reshape(1, 1, -1)
    prompt_speech_emb = torch.zeros(1, 0, 1024, dtype=text_encoded.dtype)

    lm_input = torch.cat([sos_emb, emb, text_encoded, task_id_emb, prompt_speech_emb], dim=1)

    x_tt = ttnn.from_torch(lm_input, layout=ttnn.TILE_LAYOUT, device=tt_llm.device)
    att_mask = torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), dtype=torch.bool))
    y_pred, _ = tt_llm.llm_encoder.forward_chunk(x_tt, offset=0, att_cache=None, att_mask=att_mask)
    y_last = ttnn.to_torch(y_pred[:, -1]).float()
    return tt_llm.llm_decoder(y_last)


@torch.inference_mode()
def _ref_inference_full(ref_llm, text, text_len, embedding, num_tokens=5):
    """Greedy decode `num_tokens` from the reference using the same `forward_chunk` path the official `inference()` uses."""
    text_emb = ref_llm.text_embedding(text)
    encoder_out, _ = ref_llm.text_encoder(text_emb, text_len, decoding_chunk_size=1, num_decoding_left_chunks=-1)
    encoder_out = ref_llm.text_encoder_affine_layer(encoder_out)

    emb = torch.nn.functional.normalize(embedding, dim=1)
    emb = ref_llm.spk_embed_affine_layer(emb).unsqueeze(1)

    sos_emb = ref_llm.llm_embedding.weight[ref_llm.sos].reshape(1, 1, -1)
    task_id_emb = ref_llm.llm_embedding.weight[ref_llm.task_id].reshape(1, 1, -1)
    lm_input = torch.cat([sos_emb, emb, encoder_out, task_id_emb], dim=1)

    generated = []
    att_cache = torch.zeros(0, 0, 0, 0)
    cnn_cache = torch.zeros(0, 0, 0, 0)
    offset = 0
    for _ in range(num_tokens):
        att_mask = torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), dtype=torch.bool))
        out, att_cache, cnn_cache = ref_llm.llm.forward_chunk(
            lm_input,
            offset=offset,
            required_cache_size=-1,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            att_mask=att_mask,
        )
        logit = ref_llm.llm_decoder(out[:, -1])
        tok = _greedy(logit.squeeze(0))
        if tok == ref_llm.eos_token:
            break
        generated.append(tok)
        offset += lm_input.shape[1]
        lm_input = ref_llm.speech_embedding.weight[tok].reshape(1, 1, -1)
    return generated


@torch.inference_mode()
def _tt_inference_full(tt_llm, text, text_len, embedding, num_tokens=5):
    """Greedy decode `num_tokens` from the TT LLM and return the list of generated token ids."""
    prompt_text = torch.zeros(1, 0, dtype=torch.int32)
    prompt_speech = torch.zeros(1, 0, dtype=torch.int32)
    tokens = []
    for tok in tt_llm.inference(
        text=text,
        text_len=text_len,
        prompt_text=prompt_text,
        prompt_text_len=torch.tensor([0], dtype=torch.int32),
        prompt_speech_token=prompt_speech,
        prompt_speech_token_len=torch.tensor([0], dtype=torch.int32),
        embedding=embedding,
        sampling=1,
        max_token_text_ratio=20.0,
        min_token_text_ratio=0.0,
    ):
        tokens.append(int(tok))
        if len(tokens) >= num_tokens:
            break
    return tokens


@pytest.fixture(scope="module")
def reference_model():
    return CosyVoiceReferenceModel(model_dir=MODEL_DIR)


def _build_batch():
    torch.manual_seed(0)
    text_token = torch.randint(100, 50000, (1, 8), dtype=torch.int32)
    text_len = torch.tensor([8], dtype=torch.int32)
    embedding = torch.randn(1, 192)
    return text_token, text_len, embedding


def test_llm_inference_first_token_pcc(device, reference_model):
    """First-token logit distribution: TT inference path vs reference forward_chunk path."""
    ref_llm = reference_model.llm
    ref_llm.eval()
    text_token, text_len, embedding = _build_batch()

    ref_logit = _ref_first_token_logits(ref_llm, text_token, text_len, embedding)

    config = create_model_config(batch_size=1, hidden_size=1024)
    tt_llm = TtCosyVoiceLLM(device, config, args=None, state_dict=ref_llm.state_dict())
    tt_logit = _tt_first_token_logits(tt_llm, text_token, text_len, embedding)

    pcc_ok, pcc_val = comp_pcc(ref_logit.float(), tt_logit.float(), pcc=0.95)
    print(f"first-token logit PCC: {pcc_val:.4f}")
    assert pcc_ok, f"first-token logit PCC {pcc_val:.4f} < 0.95"
    assert int(ref_logit.argmax(-1).item()) == int(tt_logit.argmax(-1).item()), (
        f"first-token argmax mismatch: ref={int(ref_logit.argmax(-1).item())}, " f"tt={int(tt_logit.argmax(-1).item())}"
    )


def test_llm_inference_greedy_matches_reference(device, reference_model):
    """Greedy decode first 5 tokens; expect first-token match (bf16 error accumulation causes later divergence)."""
    ref_llm = reference_model.llm
    ref_llm.eval()
    text_token, text_len, embedding = _build_batch()

    config = create_model_config(batch_size=1, hidden_size=1024)
    tt_llm = TtCosyVoiceLLM(device, config, args=None, state_dict=ref_llm.state_dict())

    ref_tokens = _ref_inference_full(ref_llm, text_token, text_len, embedding, num_tokens=5)
    tt_tokens = _tt_inference_full(tt_llm, text_token, text_len, embedding, num_tokens=5)

    print(f"ref_tokens={ref_tokens}")
    print(f"tt_tokens={tt_tokens}")
    assert len(ref_tokens) > 0 and len(tt_tokens) > 0
    matches = sum(1 for r, t in zip(ref_tokens, tt_tokens) if r == t)
    match_rate = matches / max(len(ref_tokens), len(tt_tokens))
    print(f"token match rate: {matches}/{max(len(ref_tokens), len(tt_tokens))} = {match_rate:.2%}")
    assert ref_tokens[0] == tt_tokens[0], f"first greedy token mismatch: ref={ref_tokens[0]}, tt={tt_tokens[0]}"


def test_llm_inference_runs_without_error(device, reference_model):
    """Smoke test: TT inference generator yields at least one valid speech token."""
    ref_llm = reference_model.llm
    ref_llm.eval()
    text_token, text_len, embedding = _build_batch()

    config = create_model_config(batch_size=1, hidden_size=1024)
    tt_llm = TtCosyVoiceLLM(device, config, args=None, state_dict=ref_llm.state_dict())

    prompt_text = torch.zeros(1, 0, dtype=torch.int32)
    prompt_speech = torch.zeros(1, 0, dtype=torch.int32)
    got = []
    for tok in tt_llm.inference(
        text=text_token,
        text_len=text_len,
        prompt_text=prompt_text,
        prompt_text_len=torch.tensor([0], dtype=torch.int32),
        prompt_speech_token=prompt_speech,
        prompt_speech_token_len=torch.tensor([0], dtype=torch.int32),
        embedding=embedding,
        sampling=5,
        max_token_text_ratio=5.0,
        min_token_text_ratio=0.0,
    ):
        got.append(int(tok))
        if len(got) >= 3:
            break

    print(f"first tokens: {got}")
    assert len(got) > 0, "inference yielded no tokens"
    assert all(0 <= t < 4096 for t in got), f"got out-of-range tokens: {got}"
