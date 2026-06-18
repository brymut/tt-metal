# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from huggingface_hub import snapshot_download


def download_model():
    model_dir = "pretrained_models/CosyVoice-300M"
    if not os.path.exists(model_dir):
        print("Downloading CosyVoice-300M...")
        snapshot_download("FunAudioLLM/CosyVoice-300M", local_dir=model_dir)
        print("Download complete.")
    else:
        print("Model already downloaded.")


if __name__ == "__main__":
    download_model()
