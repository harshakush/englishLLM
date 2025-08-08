import torch
import json
import os
from transformers import PretrainedConfig

# Update this import to your actual model file if needed
from englishLLM import CustomTransformerConfig, CustomTransformerLM

OUTPUT_DIR = "/home/harsha/Dev/englishLLM/output"
HF_DIR = os.path.join(OUTPUT_DIR, "hf_model")
SEQ_LEN = 128  # Set this to your model's sequence length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open(os.path.join(HF_DIR, "vocab.json"), "r", encoding="utf-8") as f:
    word2idx = json.load(f)

# Auto-create idx2word.json if missing
idx2word_path = os.path.join(HF_DIR, "idx2word.json")
if not os.path.exists(idx2word_path):
    idx2word = {int(v): k for k, v in word2idx.items()}
    with open(idx2word_path, "w", encoding="utf-8") as f:
        json.dump(idx2word, f, ensure_ascii=False, indent=2)
    print("idx2word.json created automatically!")
else:
    with open(idx2word_path, "r", encoding="utf-8") as f:
        idx2word = {int(k): v for k, v in json.load(f).items()}

# Load model config and weights
config = CustomTransformerConfig.from_pretrained(HF_DIR)
model = CustomTransformerLM(config)
model.model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_transformer_lm.pt"), map_location=device))
model = model.to(device)
model.eval()

def generate(seed, max_new_tokens=30, temperature=0.8):
    tokens = [word2idx.get(w, word2idx['<unk>']) for w in seed]
    for _ in range(max_new_tokens):
        input_tokens = tokens[-SEQ_LEN:] if len(tokens) >= SEQ_LEN else ([word2idx['<pad>']] * (SEQ_LEN - len(tokens)) + tokens)
        x = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
        tokens.append(next_token)
        if idx2word[next_token] in ['.', '!', '?']:
            break
    return ' '.join([idx2word[t] for t in tokens])

if __name__ == "__main__":
    print("=== Transformer LM Prompt Generation ===")
    print("Type your prompt (words separated by spaces). Ctrl+C to exit.")
    while True:
        try:
            user_input = input("\nPrompt: ").strip()
            if not user_input:
                continue
            seed = user_input.lower().split()
            try:
                max_new = int(input("Max new tokens [30]: ").strip() or 30)
            except Exception:
                max_new = 30
            try:
                temp = float(input("Temperature [0.8]: ").strip() or 0.8)
            except Exception:
                temp = 0.8
            output = generate(seed, max_new_tokens=max_new, temperature=temp)
            print("\nGenerated:", output)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break