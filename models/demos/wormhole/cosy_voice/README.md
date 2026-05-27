# CosyVoice TTS on Wormhole (N300)

## Platforms
    Wormhole (N300)

## Introduction
CosyVoice is a multi-lingual large voice generation model from Alibaba's FunAudioLLM that provides
full-stack TTS capabilities. This implementation runs the CosyVoice2-0.5B model on Tenstorrent
hardware using TTNN APIs for high-throughput, low-latency multilingual speech synthesis.

### Architecture
The model consists of a 3-stage pipeline:
1. **LLM Backbone** (Qwen2-0.5B): Autoregressive semantic token generation from text
2. **Flow Matching Decoder** (Chunk-aware CFM): Converts semantic tokens to mel spectrograms
3. **HiFi-GAN Vocoder**: Reconstructs audio waveform from mel spectrograms

### Supported Inference Modes
- **SFT**: Generate speech with predefined speakers
- **Zero-shot**: Generate speech with reference audio (voice cloning)
- **Cross-lingual**: Generate speech in different language from reference
- **Instruct**: Generate expressive speech with natural language instructions

### Supported Languages
Chinese, English, Japanese, Cantonese, Korean

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- CosyVoice dependencies: `pip install hyperpyyaml onnxruntime conformer`

### Download Model Weights
```python
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='models/demos/wormhole/cosy_voice/pretrained_models/CosyVoice2-0.5B')
```

## How to Run

### Quick Demo
```bash
pytest --disable-warnings -q -s models/demos/wormhole/cosy_voice/demo/demo.py::test_cosyvoice_demo
```

### Run with Custom Text
```bash
pytest --disable-warnings -q -s models/demos/wormhole/cosy_voice/demo/demo.py::test_cosyvoice_demo --text="Hello world"
```

## Testing

### Unit Tests
Navigate to the `tt-metal` directory:
```bash
cd tt-metal
```

#### RMS Norm
```bash
pytest -svv models/demos/wormhole/cosy_voice/tests/test_ttnn_rms_norm.py
```

#### Attention (GQA + RoPE)
```bash
pytest -svv models/demos/wormhole/cosy_voice/tests/test_ttnn_attention.py
```

#### MLP
```bash
pytest -svv models/demos/wormhole/cosy_voice/tests/test_ttnn_mlp.py
```

#### Decoder Layer
```bash
pytest -svv models/demos/wormhole/cosy_voice/tests/test_ttnn_decoder_layer.py
```

#### Full LLM
```bash
pytest -svv models/demos/wormhole/cosy_voice/tests/test_ttnn_llm.py
```

#### Flow Matching
```bash
pytest -svv models/demos/wormhole/cosy_voice/tests/test_ttnn_flow_matching.py
```

#### HiFi-GAN Vocoder
```bash
pytest -svv models/demos/wormhole/cosy_voice/tests/test_ttnn_hifigan.py
```

#### Full Pipeline
```bash
pytest -svv models/demos/wormhole/cosy_voice/tests/test_ttnn_full_pipeline.py
```

## Performance Targets (Stage 1)
| Metric | Target |
|---|---|
| Semantic token generation | ≥ 30 tokens/sec |
| Real-time factor (RTF) | < 0.5 |
| Token accuracy vs PyTorch | > 95% |
| WER | < 3.0 |
| Speaker similarity | > 60 |

## References
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice2 Paper](https://arxiv.org/abs/2412.10117)
- [CosyVoice3 Paper](https://arxiv.org/abs/2505.17589)
- [TTNN Model Bringup Guide](../../tech_reports/ttnn/TTNN-model-bringup.md)
