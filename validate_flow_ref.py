import sys, os
from pathlib import Path

sys.path.insert(0, str(Path("models/demos/wormhole/cosyvoice/reference/CosyVoice")))
sys.path.insert(0, str(Path("models/demos/wormhole/cosyvoice/reference/CosyVoice/third_party/Matcha-TTS")))

import torch
from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel

model_dir = os.getenv("COSYVOICE_MODEL_DIR", "/root/tt-metal/pretrained_models/CosyVoice-300M")
print("Loading reference model...")
ref = CosyVoiceReferenceModel(model_dir=model_dir)
print("Reference model loaded.")

flow = ref.flow
device = torch.device("cpu")

# Minimal synthetic inputs
batch_size = 1
token_len2 = 5
token_len1 = 5
mel_len1 = 10

token = torch.randint(0, 4096, (batch_size, token_len2), dtype=torch.int32)
token_len = torch.tensor([token_len2], dtype=torch.int32)
prompt_token = torch.randint(0, 4096, (batch_size, token_len1), dtype=torch.int32)
prompt_token_len = torch.tensor([token_len1], dtype=torch.int32)
prompt_feat = torch.randn(batch_size, mel_len1, flow.output_size, dtype=torch.float32)
prompt_feat_len = torch.tensor([mel_len1], dtype=torch.int32)
embedding = torch.randn(batch_size, 192, dtype=torch.float32)

# ConditionalCFM expects a cache tensor, not None
flow_cache = torch.zeros(batch_size, 80, 0, 2, dtype=torch.float32)

print("Running flow.inference...")
with torch.no_grad():
    feat, flow_cache = flow.inference(
        token=token,
        token_len=token_len,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=prompt_feat_len,
        embedding=embedding,
        flow_cache=flow_cache,
    )

print(f"feat shape: {feat.shape}")
print(f"feat dtype: {feat.dtype}")
print(f"feat mean: {feat.mean().item():.6f}, std: {feat.std().item():.6f}")
print("Test passed.")
