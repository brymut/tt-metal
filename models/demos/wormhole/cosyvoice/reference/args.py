# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum


class ModelMode(IntEnum):
    DECODE = 0
    PREFILL = 1


@dataclass
class CosyVoiceArgs:
    llm_input_size: int = 896
    llm_output_size: int = 896
    text_token_size: int = 8388
    speech_token_size: int = 8192
    spk_embed_dim: int = 192
    mode: ModelMode = ModelMode.DECODE
