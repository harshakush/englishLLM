import os
import json
from englishLLM import CustomTransformerConfig, CustomTransformerLM
# Path to your HuggingFace-format model directory
checkpoint_path = "/home/harsha/Dev/englishLLM/output/hf_model"

# Load config and model using your custom classes
config = CustomTransformerConfig.from_pretrained(checkpoint_path)
model = CustomTransformerLM.from_pretrained(checkpoint_path, config=config)

# (Optional) If you have a vocab.json, you can load it here
with open(os.path.join(checkpoint_path, "vocab.json"), "r", encoding="utf-8") as f:
    word2idx = json.load(f)

# Save model and config to a new directory if you want
output_dir = "hf_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
config.save_pretrained(output_dir)

# Save vocab.json if needed
with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(word2idx, f, ensure_ascii=False, indent=2)

print(f"Model, config, and vocab saved to {output_dir}")