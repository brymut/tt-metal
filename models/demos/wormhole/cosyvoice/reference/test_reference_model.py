# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel


def test_load_reference():
    model_dir = "pretrained_models/CosyVoice-300M"
    ref_model = CosyVoiceReferenceModel(model_dir=model_dir)
    print("Successfully loaded CosyVoiceReferenceModel!")
    print(f"LLM type: {type(ref_model.llm)}")
    print(f"Flow type: {type(ref_model.flow)}")
    print(f"HiFiGAN type: {type(ref_model.hifigan)}")


if __name__ == "__main__":
    test_load_reference()
