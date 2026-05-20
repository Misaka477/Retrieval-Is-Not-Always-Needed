"""Save TinyStories BPE tokenizer for reuse."""
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from tqdm import tqdm

print("Loading 20000 stories...", flush=True)
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
texts = []
for i, item in tqdm(enumerate(ds), desc="Loading", total=20000):
    if i >= 20000: break
    texts.append(item["text"])

print("Training BPE tokenizer...", flush=True)
tok = ByteLevelBPETokenizer()
tok.train_from_iterator(texts, vocab_size=4096, min_frequency=2,
                        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
tok.save_model("D:\\Software_Development\\Project\\RINA_Project\\checkpoints", "ts")
print(f"Saved: vocab={tok.get_vocab_size()}", flush=True)
