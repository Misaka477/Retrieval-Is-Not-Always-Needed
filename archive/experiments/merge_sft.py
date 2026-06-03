import json, os, random
random.seed(42)
sft_dir = "experiments/checkpoints/sft_data"
files = ["r1_distill.jsonl", "o1_cot.jsonl", "magic_code.jsonl", "gsm8k.jsonl"]
all_data = []
for f in files:
    path = os.path.join(sft_dir, f)
    if not os.path.exists(path):
        print(f"Missing: {f} — rerun prepare_sft_data.py")
        exit(1)
    with open(path, encoding="utf-8") as fp:
        data = [json.loads(l) for l in fp]
    all_data.extend(data)
    print(f"{f}: {len(data)}")
random.shuffle(all_data)
out = os.path.join(sft_dir, "sft_all.jsonl")
with open(out, "w", encoding="utf-8") as fp:
    for item in all_data:
        fp.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"\nMerged: {len(all_data)} examples → {out}")
