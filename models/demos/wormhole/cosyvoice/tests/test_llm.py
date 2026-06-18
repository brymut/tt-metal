# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_llm import TtCosyVoiceLLM
from models.demos.wormhole.cosyvoice.tt.model_config import create_model_config

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = os.getenv(
    "COSYVOICE_MODEL_DIR", str(PROJECT_ROOT.parent.parent.parent.parent) + "/pretrained_models/CosyVoice-300M"
)


@pytest.fixture(scope="module")
def reference_model():
    model_dir = os.getenv("COSYVOICE_MODEL_DIR", MODEL_DIR)
    return CosyVoiceReferenceModel(model_dir=model_dir)


def test_cosyvoice_llm(device, reference_model):
    # Setup test for LLM backbone

    # 1. Get reference LLM
    ref_llm = reference_model.llm
    ref_llm.eval()

    # 2. Setup mock input for testing the LLM (batch_size=1)
    batch = {
        "text_token": torch.randint(0, 51866, (1, 10)),
        "text_token_len": torch.tensor([10], dtype=torch.int32),
        "speech_token": torch.randint(0, 4096, (1, 20)),
        "speech_token_len": torch.tensor([20], dtype=torch.int32),
        "embedding": torch.randn(1, 192),
    }

    # 3. Get reference output (capture logits by monkey-patching criterion)
    class FakeCriterion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = None

        def forward(self, logits, target):
            self.logits = logits.detach()
            return logits.mean()

    fake_criterion = FakeCriterion()
    original_criterion = ref_llm.criterion_ce
    ref_llm.criterion_ce = fake_criterion

    with torch.no_grad():
        ref_output = ref_llm(batch, device=torch.device("cpu"))

    ref_llm.criterion_ce = original_criterion
    ref_logits = fake_criterion.logits

    # 4. Setup TTNN Model
    config = create_model_config(batch_size=1, hidden_size=1024)
    tt_llm = TtCosyVoiceLLM(device, config, args=None, state_dict=ref_llm.state_dict())

    # 5. Get TTNN output
    tt_output = tt_llm(batch)

    # 6. Compare shapes
    tt_logits = tt_output["logits"]
    assert ref_logits.shape == tt_logits.shape, f"Shape mismatch: ref={ref_logits.shape}, tt={tt_logits.shape}"

    # 7. PCC comparison
    # Baseline PCC before device attention was ~0.9848.
    pcc_result, pcc_value = comp_pcc(ref_logits, tt_logits, pcc=0.98)
    print(f"Reference LLM output keys: {ref_output.keys()}")
    print(f"Logits PCC: {pcc_value}")
    assert pcc_result, f"PCC check failed: {pcc_value} < 0.98"
