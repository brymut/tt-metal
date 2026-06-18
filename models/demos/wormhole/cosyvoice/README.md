# CosyVoice-300M TTNN Demo

## Platforms:
Wormhole (N300, T3000)

## Introduction
This demo runs the CosyVoice-300M multi-lingual large voice generation model on Tenstorrent hardware using the TTNN API. It provides end-to-end TTS capabilities (semantic token generation, acoustic modeling, and waveform generation) with multiple inference modes including SFT, zero-shot, cross-lingual, and instruct-based.

## Prerequisites
- Download the official `FunAudioLLM/CosyVoice-300M` model weights from HuggingFace.
- Install the `tt-metal` software stack.

## How to Run

To run the full end-to-end demo on a sample prompt:
```bash
pytest -svv models/demos/wormhole/cosyvoice/demo/demo.py
```

## Testing

### Unit Tests
To run unit tests for the individual modules (LLM, Flow, HiFi-GAN):
```bash
pytest -svv models/demos/wormhole/cosyvoice/tests/test_llm.py
pytest -svv models/demos/wormhole/cosyvoice/tests/test_flow.py
pytest -svv models/demos/wormhole/cosyvoice/tests/test_hifigan.py
```

To run end-to-end model functional test:
```bash
pytest -svv models/demos/wormhole/cosyvoice/tests/test_cosyvoice_model.py
```

### Performance Tests
To evaluate performance targets (>30 tokens/sec, RTF < 0.5):
```bash
pytest -svv models/demos/wormhole/cosyvoice/tests/test_perf.py
```
