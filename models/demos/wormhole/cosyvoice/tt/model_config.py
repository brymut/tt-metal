# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.cosyvoice.reference.args import ModelMode


def create_model_config(batch_size, hidden_size, mode=ModelMode.DECODE, seq_len=1):
    configs = {}
    configs["mode"] = mode
    configs["seq_len"] = seq_len
    configs["batch_size"] = batch_size

    if mode == ModelMode.DECODE:
        configs["dtype"] = {"activations": ttnn.bfloat8_b, "weights": ttnn.bfloat4_b}
    else:
        configs["dtype"] = {"activations": ttnn.bfloat16, "weights": ttnn.bfloat8_b}

    return configs
