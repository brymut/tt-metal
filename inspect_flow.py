import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path("models/demos/wormhole/cosyvoice/reference/CosyVoice")))
sys.path.insert(0, str(Path("models/demos/wormhole/cosyvoice/reference/CosyVoice/third_party/Matcha-TTS")))

from models.demos.wormhole.cosyvoice.reference.model import CosyVoiceReferenceModel

model_dir = os.getenv("COSYVOICE_MODEL_DIR", "/root/tt-metal/pretrained_models/CosyVoice-300M")
ref = CosyVoiceReferenceModel(model_dir=model_dir)

flow = ref.flow

# Flow encoder details
encoder = flow.encoder
print("=== Flow Encoder ===")
print(f"type = {type(encoder)}")
print(f"num_blocks = {len(encoder.encoders)}")
print(f"output_size = {encoder.output_size()}")

# Length regulator details
lr = flow.length_regulator
print("\n=== Length Regulator ===")
print(f"type = {type(lr)}")
print(f"model = {lr.model}")

# Decoder (ConditionalCFM) details
decoder = flow.decoder
print("\n=== ConditionalCFM ===")
print(f"type = {type(decoder)}")
print(f"estimator = {decoder.estimator}")
print(f"estimator type = {type(decoder.estimator)}")

# Estimator details
est = decoder.estimator
print("\n=== Estimator (UNet) ===")
print(f"type = {type(est)}")
print(f"in_channels = {est.in_channels if hasattr(est, 'in_channels') else 'N/A'}")
print(f"out_channels = {est.out_channels if hasattr(est, 'out_channels') else 'N/A'}")
print(f"down_blocks len = {len(est.down_blocks)}")
print(f"mid_blocks len = {len(est.mid_blocks)}")
print(f"up_blocks len = {len(est.up_blocks)}")

# Count parameters per submodule
sd = flow.state_dict()
submodule_counts = {}
for k in sd.keys():
    top = k.split(".")[0]
    submodule_counts[top] = submodule_counts.get(top, 0) + 1
print("\n=== Parameter counts per top-level submodule ===")
for k, v in submodule_counts.items():
    print(f"  {k}: {v} tensors")

# Full state dict keys
with open("/root/tt-metal/flow_state_dict_keys.txt", "w") as f:
    for k in sd.keys():
        f.write(f"{k}: {list(sd[k].shape)}\n")
print("\nSaved full key list to /root/tt-metal/flow_state_dict_keys.txt")
