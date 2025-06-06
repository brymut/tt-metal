# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov4.common import load_torch_model
from models.demos.yolov4.tt.downsample3 import Down3
from models.demos.yolov4.tt.model_preprocessing import create_ds3_model_parameters
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_down3(device, reset_seeds, model_location_generator, resolution):
    torch.manual_seed(0)

    torch_model = load_torch_model(model_location_generator, module="down3")
    if resolution == (320, 320):
        torch_input = torch.randn((1, 128, 80, 80), dtype=torch.float)
    elif resolution == (640, 640):
        torch_input = torch.randn((1, 128, 160, 160), dtype=torch.float)
    else:
        raise ValueError(f"Unsupported resolution {resolution}")
    ref = torch_model(torch_input)

    parameters = create_ds3_model_parameters(torch_model, torch_input, resolution, device)

    ttnn_model = Down3(device, parameters, parameters.conv_args)

    torch_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)

    result_ttnn = ttnn_model(ttnn_input)

    start_time = time.time()
    for x in range(2):
        result_ttnn = ttnn_model(ttnn_input)
    logger.info(f"Time taken: {time.time() - start_time}")

    result = ttnn.to_torch(result_ttnn)
    result = result.permute(0, 3, 1, 2)
    result = result.reshape(ref.shape)
    assert_with_pcc(result, ref, 0.96)  # PCC 0.96 - The PCC will improve once #3612 is resolved.
