"""Filter SFT data: length filter + identity replacement."""
import json, os, random

SRC = "experiments/checkpoints/sft_data/sft_all.jsonl"
DST = "experiments/checkpoints/sft_data/sft_filtered.jsonl"
MIN_TOKENS, MAX_TOKENS = 64, 1024
random.seed(42)

# identity patterns to replace
MODEL_NAMES = [
    "ChatGPT", "GPT", "Claude", "Gemini",
    "Qwen", "QWen", "DeepSeek", "LLaMA", "Llama",
    "Mistral", "Mixtral", "Gemma", "Phi",
    "Baichuan", "Yi", "GLM", "ChatGLM",
    "ERNIE", "Spark", "Tongyi", "Doubao", "Kimi",
    "Step", "MiniMax", "Hunyuan", "Copilot", "Bard",
]
IDENTITY_PATTERNS = [
    "I am an AI language model", "I'm an AI language model",
    "I am a large language model", "I'm a large language model",
    "I am an AI assistant", "I'm an AI assistant",
    "I am an AI", "I'm an AI",
    "As a large language model", "As an AI language model",
    "As an AI assistant", "As an AI",
    "as a large language model", "as an AI language model",
    "as an AI assistant", "as an AI",
    "I am a helpful assistant", "I'm a helpful assistant",
    "I was created by", "I am here to help",
    "I am developed by", "I am built by",
    "I was trained by",
]
for n in MODEL_NAMES:
    IDENTITY_PATTERNS.append(f"I am {n}")
    IDENTITY_PATTERNS.append(f"I'm {n}")
    IDENTITY_PATTERNS.append(f"我是{n}")
    IDENTITY_PATTERNS.append(f"我是AI{n}")


def clean(text):
    for p in IDENTITY_PATTERNS:
        text = text.replace(p, "I am Anthelia")
    return text


def estimate_tokens(text):
    return len(text.encode("utf-8")) // 4  # rough estimate


with open(SRC, encoding="utf-8") as f:
    raw = [json.loads(l) for l in f]
print(f"Loaded: {len(raw)} examples")

filtered = []
skipped = {"too_short": 0, "too_long": 0, "no_content": 0}
for item in raw:
    text = item.get("text", "")
    if len(text.strip()) < 20:
        skipped["no_content"] += 1
        continue
    nt = estimate_tokens(text)
    if nt < MIN_TOKENS:
        skipped["too_short"] += 1
        continue
    if nt > MAX_TOKENS:
        skipped["too_long"] += 1
        continue
    text = clean(text)
    filtered.append({"text": text})

print(f"Filtered: {len(filtered)}")
for k, v in skipped.items():
    print(f"  {k}: {v}")

random.shuffle(filtered)
with open(DST, "w", encoding="utf-8") as f:
    for item in filtered:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved: {DST}")
