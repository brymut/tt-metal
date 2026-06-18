# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test config: TTNN device fixture + CLI flags for the audio eval harness."""

import pytest

import ttnn


def pytest_addoption(parser):
    parser.addoption(
        "--eval-modes", nargs="*", default=None, help="Restrict the audio eval to these modes (default: all)"
    )
    parser.addoption(
        "--eval-languages", nargs="*", default=None, help="Restrict the audio eval to these languages (default: all)"
    )


@pytest.fixture(scope="session")
def device():
    """Create one shared TTNN device for the test session.

    The UNet port requires a larger L1 than the default (the flow UNet has
    ~1M conv weights and creates many sharded intermediate tensors). 64 KB
    matches what other conv-heavy demos use.
    """
    dev = ttnn.open_device(
        device_id=0,
        l1_small_size=64 << 10,
        trace_region_size=128 << 20,
    )
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)
