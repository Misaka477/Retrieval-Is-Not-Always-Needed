"""SFT data preparation: R1 + o1 CoT + Magic Code + GSM8K + OpenHermes.
Converts to unified format, replaces identity tags, saves as jsonl."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "sft_data")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)


def save(data, name):
    path = os.path.join(OUT_DIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(data)} examples -> {path}")


def replace_identity(text):
    patterns = [
        "I am an AI language model", "I'm an AI language model",
        "I am a large language model", "I'm a large language model",
        "I am an AI assistant", "I'm an AI assistant",
        "I am an AI", "I'm an AI",
        "As a large language model", "As an AI language model",
        "As an AI assistant", "As an AI",
        "as a large language model", "as an AI language model",
        "as an AI assistant", "as an AI",
        "I am ChatGPT", "I'm ChatGPT",
        "I am GPT", "I'm GPT",
        "I am Claude", "I'm Claude",
        "I am a helpful assistant", "I'm a helpful assistant",
        "I was created by", "I am here to help",
    ]
    for p in patterns:
        text = text.replace(p, "I am Anthelia")
    return text


def format_chat(inst, out=""):
    text = f"<|user|>\n{inst}\n<|assistant|>\n{out}"
    return replace_identity(text)


# 1. R1 蒸馏 (local)
print("[1/6] R1 distillation...")
r1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "R1_110k", "train_filtered_2048.jsonl")
r1_data = []
with open(r1_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = format_chat(item.get("instruction", ""), item.get("output", ""))
        r1_data.append({"text": text})
save(r1_data, "r1_distill")

# 2. o1 CoT
print("[2/6] O1-OPEN/OpenO1-SFT...")
o1_raw = load_dataset("O1-OPEN/OpenO1-SFT", split="train", streaming=True)
o1_data = []
for i, item in enumerate(o1_raw):
    text = format_chat(item["instruction"], item["output"])
    o1_data.append({"text": text})
    if len(o1_data) >= 50000:
        break
save(o1_data, "o1_cot")

# 3. Magic Code
print("[3/6] Magicoder-OSS-Instruct-75K...")
code_raw = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
code_data = []
for i, item in enumerate(code_raw):
    text = format_chat(item["problem"], item["solution"])
    code_data.append({"text": text})
    if len(code_data) >= 50000:
        break
save(code_data, "magic_code")

# 4. GSM8K (math reasoning)
print("[4/6] GSM8K...")
gsm_raw = load_dataset("gsm8k", "main", split="train", streaming=True)
gsm_data = []
for i, item in enumerate(gsm_raw):
    text = format_chat(item["question"], item["answer"])
    gsm_data.append({"text": text})
save(gsm_data, "gsm8k")

# 5. OpenHermes (general instruction) — skipped, download issue
hermes_data = []

# 6. Merge & shuffle
print("[6/6] Merging...")
all_data = r1_data + o1_data + code_data + gsm_data
random.shuffle(all_data)
save(all_data, "sft_all")
print(f"\nTotal: {len(all_data)} examples")
print(f"Composition: R1={len(r1_data)}, o1={len(o1_data)}, Code={len(code_data)}, GSM8K={len(gsm_data)}")
