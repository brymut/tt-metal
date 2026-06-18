# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import torch.nn as nn


class CosyVoiceReferenceModel(nn.Module):
    def __init__(self, model_dir):
        super().__init__()

        # Add CosyVoice to sys.path to allow imports
        cosyvoice_path = Path(__file__).parent / "CosyVoice"
        if str(cosyvoice_path) not in sys.path:
            sys.path.append(str(cosyvoice_path))
        matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"
        if str(matcha_path) not in sys.path:
            sys.path.append(str(matcha_path))

        from cosyvoice.cli.cosyvoice import AutoModel

        self.cosyvoice = AutoModel(model_dir=model_dir)
        self.model = self.cosyvoice.model

        self.llm = self.model.llm
        self.flow = self.model.flow
        self.hifigan = self.model.hift

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
