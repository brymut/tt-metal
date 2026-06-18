# Reference Golden Path Artifacts

This directory contains the PyTorch reference E2E pipeline outputs (LLM → Flow → HiFi-GAN)
for a fixed set of `(mode, language)` test cases. These are the ground truth against
which the TTNN port will be compared once the E2E pipeline is wired.

## Layout

```
golden/
├── inputs/    # saved input tensors (text tokens, prompt tokens, prompt mel, spk embed)
├── tokens/    # saved semantic token ids (LLM output)
├── mels/      # saved mel spectrograms (flow output, used for TT flow unit tests)
├── wavs/      # saved waveform files (22050 Hz, used for WER + speaker-similarity)
└── README.md  # this file
```

## How goldens are generated

`reference/golden_pipeline.py` exposes `regenerate_goldens(model_dir, golden_dir)`.
For each `(mode, language)` test case in `reference/golden_prompts.py`, it:

1. Builds a deterministic input dict from a fixed seed (so re-runs are bit-exact).
2. Tokenizes text with the whisper multilingual tokenizer (with `allowed_special='all'`).
3. Runs `llm.inference(...)` → `flow.inference(...)` → `hift.inference(...)` on CPU.
4. Saves all inputs, intermediate tokens, mels, and the final wav to this directory.

## Test cases

Modes (4): `sft`, `zero_shot`, `cross_lingual`, `instruct`.
Languages (5): `en`, `zh`, `ja`, `yue`, `ko`.

Total: 20 test cases. Each yields one `.wav` and one mel + token file.

## Regenerate

```bash
source python_env/bin/activate
cd /root/tt-metal
python -m models.demos.wormhole.cosyvoice.reference.golden_pipeline \
    --model-dir pretrained_models/CosyVoice-300M \
    --golden-dir models/demos/wormhole/cosyvoice/tests/golden
```

Expected runtime: ~5-10 minutes on CPU (LLM + 10-step flow + HiFi-GAN per case).

## Why no real prompt audio

The CosyVoice-300M base checkpoint ships without `spk2info.pt` or sample
`asset/zero_shot_prompt.wav` files. To avoid downloading extra audio assets,
the golden path uses **fixed-seed synthetic tensors** for `prompt_speech_token`,
`prompt_speech_feat`, and `spk_embedding`. This:

- Is fully deterministic (re-runs produce bit-exact wavs).
- Exercises the full pipeline (LLM → flow → vocoder) end-to-end.
- Documents the exact tensor shape / dtype contract the TTNN port must honor.

A future task can add a separate `real_prompt_audio/` subdir with .wavs
downloaded from the official CosyVoice repo for higher-fidelity reference.
