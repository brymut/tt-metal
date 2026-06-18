#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch

import ttnn

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = str(PROJECT_ROOT.parent.parent.parent.parent) + "/pretrained_models/CosyVoice-300M"

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel
from models.demos.wormhole.cosyvoice.tt.cosyvoice_flow import TtFlowEncoder


def main():
    ref_model = CosyVoiceReferenceModel(model_dir=MODEL_DIR)
    ref_flow = ref_model.flow

    # Synthetic input matching the flow encoder's expected input
    batch_size = 1
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 512, dtype=torch.float32)

    # Reference flow encoder output
    ref_flow.eval()
    with torch.no_grad():
        ref_out, ref_mask = ref_flow.encoder(x, torch.tensor([seq_len], dtype=torch.int32))

    # TT flow encoder output
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    tt_encoder = TtFlowEncoder(device, state_dict=ref_flow.state_dict())
    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out, tt_mask = tt_encoder(x_tt, token_len=torch.tensor([seq_len], dtype=torch.int32))
    tt_out_torch = ttnn.to_torch(tt_out).float()

    print(f"Reference output shape: {ref_out.shape}")
    print(f"TT output shape: {tt_out_torch.shape}")
    print(f"Reference output mean: {ref_out.mean():.6f}, std: {ref_out.std():.6f}")
    print(f"TT output mean: {tt_out_torch.mean():.6f}, std: {tt_out_torch.std():.6f}")

    pcc_result, pcc_value = comp_pcc(ref_out, tt_out_torch, pcc=0.0)
    print(f"Flow encoder PCC: {pcc_value:.6f}")

    diff = (ref_out - tt_out_torch).abs()
    print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
