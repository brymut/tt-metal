# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_llm import TtCosyVoiceLLM
from models.demos.wormhole.cosyvoice.tt.model_config import create_model_config

MODEL_DIR = "pretrained_models/CosyVoice-300M"


def test_debug_pcc():
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_llm = ref_model.llm
    ref_llm.eval()

    batch = {
        "text_token": torch.randint(0, 51866, (1, 10)),
        "text_token_len": torch.tensor([10], dtype=torch.int32),
        "speech_token": torch.randint(0, 4096, (1, 20)),
        "speech_token_len": torch.tensor([20], dtype=torch.int32),
        "embedding": torch.randn(1, 192),
    }

    # Reference forward (capture logits)
    class FakeCriterion(torch.nn.Module):
        def forward(self, logits, target):
            FakeCriterion.logits = logits.detach()
            return logits.mean()

    original_criterion = ref_llm.criterion_ce
    ref_llm.criterion_ce = FakeCriterion()
    with torch.no_grad():
        ref_llm(batch, device=torch.device("cpu"))
    ref_llm.criterion_ce = original_criterion
    ref_logits = FakeCriterion.logits

    # TT forward
    dev = ttnn.open_device(device_id=0, trace_region_size=128 << 20)
    dev.enable_program_cache()
    config = create_model_config(batch_size=1, hidden_size=1024)
    tt_llm = TtCosyVoiceLLM(dev, config, args=None, state_dict=ref_llm.state_dict())
    tt_output = tt_llm(batch)
    tt_logits = tt_output["logits"]

    print(f"Shapes: ref={ref_logits.shape}, tt={tt_logits.shape}")
    print(f"Max diff: {(ref_logits - tt_logits).abs().max().item()}")
    print(f"Mean diff: {(ref_logits - tt_logits).abs().mean().item()}")
    pcc_ok, pcc_val = comp_pcc(ref_logits, tt_logits, pcc=0.99)
    print(f"Logits PCC: {pcc_val}")

    ttnn.close_device(dev)


if __name__ == "__main__":
    test_debug_pcc()
