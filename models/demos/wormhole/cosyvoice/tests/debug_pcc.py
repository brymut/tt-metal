#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

import ttnn

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = "pretrained_models/CosyVoice-300M"

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_llm import TtCosyVoiceLLM
from models.demos.wormhole.cosyvoice.tt.model_config import create_model_config


def comp_tensors(ref, out, name):
    if ref is None or out is None:
        print(f"[{name}] None tensor encountered")
        return False, 0.0
    if ref.shape != out.shape:
        print(f"[{name}] SHAPE MISMATCH ref={ref.shape} out={out.shape}")
        return False, 0.0
    diff = (ref - out).abs()
    print(f"[{name}] max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}")
    pcc_ok, pcc_val = comp_pcc(ref, out, pcc=0.0)  # pcc=0 to just get the value
    print(f"[{name}] PCC={pcc_val:.6f}")
    return pcc_ok, pcc_val


def main():
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_llm = ref_model.llm
    ref_llm.eval()

    torch.manual_seed(42)
    batch = {
        "text_token": torch.randint(0, 51866, (1, 10)),
        "text_token_len": torch.tensor([10], dtype=torch.int32),
        "speech_token": torch.randint(0, 4096, (1, 20)),
        "speech_token_len": torch.tensor([20], dtype=torch.int32),
        "embedding": torch.randn(1, 192),
    }

    # ---------- Reference forward (manual, up to logits) ----------
    with torch.no_grad():
        # --- Text encoder (raw, before affine) ---
        text_emb = ref_llm.text_embedding(batch["text_token"])
        text_enc_raw, text_mask = ref_llm.text_encoder(
            text_emb, batch["text_token_len"], decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        text_enc_lens = text_mask.squeeze(1).sum(1)
        text_enc = ref_llm.text_encoder_affine_layer(text_enc_raw)

        # --- Embedding ---
        emb = F.normalize(batch["embedding"], dim=1)
        emb = ref_llm.spk_embed_affine_layer(emb)
        emb = emb.unsqueeze(1)

        # --- SOS / Task ---
        sos_emb = ref_llm.llm_embedding.weight[0].reshape(1, 1, -1)
        task_id_emb = ref_llm.llm_embedding.weight[1].reshape(1, 1, -1)

        # --- Speech ---
        speech_emb = ref_llm.speech_embedding(batch["speech_token"])

        # --- Pad/Unpad ---
        lm_in, lm_in_len = ref_llm.pad_unpad_sequence(
            sos_emb, emb, text_enc, text_enc_lens, task_id_emb, speech_emb, batch["speech_token_len"]
        )

        # --- LM ---
        lm_out, _ = ref_llm.llm(lm_in, lm_in_len)
        ref_logits = ref_llm.llm_decoder(lm_out)

    # ---------- TTNN forward (with hack: bypass TT text encoder) ----------
    dev = ttnn.open_device(device_id=0, trace_region_size=128 << 20)
    dev.enable_program_cache()
    config = create_model_config(batch_size=1, hidden_size=1024)
    tt_llm = TtCosyVoiceLLM(dev, config, args=None, state_dict=ref_llm.state_dict())

    # QUICK HACK: Use REFERENCE text encoder output to isolate LLM pipeline accuracy.
    print("\n[HACK] Bypassing TT text encoder; using reference PyTorch text encoder output...")

    with torch.no_grad():
        text_emb = ref_llm.text_embedding(batch["text_token"])
        text_enc_raw_hack, text_mask = ref_llm.text_encoder(
            text_emb, batch["text_token_len"], decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
    tt_text = tt_llm.text_encoder_affine_layer(text_enc_raw_hack)
    # Alias for comparison below
    tt_text_enc = text_enc_raw_hack

    # Speech and embedding (same PyTorch code as reference)
    tt_emb = F.normalize(batch["embedding"], dim=1)
    tt_emb = tt_llm.spk_embed_affine_layer(tt_emb)
    tt_emb = tt_emb.unsqueeze(1)

    sos_emb_tt = tt_llm.llm_embedding.weight[0].reshape(1, 1, -1)
    task_id_emb_tt = tt_llm.llm_embedding.weight[1].reshape(1, 1, -1)
    speech_emb_tt = tt_llm.speech_embedding(batch["speech_token"])

    # Pad/unpad sequence (same PyTorch logic)
    text_unpadded = unpad_sequence(tt_text, batch["text_token_len"].cpu(), batch_first=True)
    speech_unpadded = unpad_sequence(speech_emb_tt, batch["speech_token_len"].cpu(), batch_first=True)

    from cosyvoice.utils.common import IGNORE_ID

    lm_input = [
        torch.concat(
            [sos_emb_tt.squeeze(dim=0), tt_emb[i], text_unpadded[i], task_id_emb_tt.squeeze(dim=0), speech_unpadded[i]],
            dim=0,
        )
        for i in range(len(text_unpadded))
    ]
    lm_input_padded = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)

    # LLM input
    lm_input_tt = ttnn.from_torch(lm_input_padded, layout=ttnn.TILE_LAYOUT, device=dev)
    lm_mask = torch.ones(1, 1, lm_input_padded.shape[1], dtype=torch.bool)
    tt_lm_out_tt, _ = tt_llm.llm_encoder(lm_input_tt, lm_mask)
    tt_lm_out = ttnn.to_torch(tt_lm_out_tt).float()

    # LLM decoder (PyTorch)
    tt_logits = tt_llm.llm_decoder(tt_lm_out)

    # ---------- Compare stage by stage ----------
    print("\n=== Text Encoder Raw ===")
    comp_tensors(text_enc_raw, tt_text_enc, "text_enc_raw")

    print("\n=== Text Enc + Affine ===")
    comp_tensors(text_enc, tt_text, "text_enc+affine")

    print("\n=== LM Input ===")
    comp_tensors(lm_in, lm_input_padded, "lm_input")

    print("\n=== LM Output ===")
    comp_tensors(lm_out, tt_lm_out, "lm_out")

    print("\n=== Final Logits ===")
    comp_tensors(ref_logits, tt_logits, "logits")

    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
