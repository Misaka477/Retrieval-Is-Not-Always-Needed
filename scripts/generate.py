"""独立生成测试: 加载最新 checkpoint, adaptive temperature + top-p + sampling."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tokenizers import Tokenizer
from rina.mohe import MoHE
from rina.sample import sample

device = "cuda"
VOCAB, DM, SEQ = 50257, 256, 64

ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../checkpoints")
ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*_resume.pt")))
if not ckpt_files:
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "mohe_large_ep*.pt")))
if not ckpt_files:
    print("No checkpoint found in", ckpt_dir)
    exit(1)

print(f"Loading: {ckpt_files[-1]}")
ckpt = torch.load(ckpt_files[-1], map_location=device, weights_only=False)

model = MoHE(VOCAB, DM, 512, n_experts=4).to(device)
sd = ckpt["model"]
for k in list(sd.keys()):
    if k.startswith("prev_route"): del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()
print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

tok = Tokenizer.from_pretrained("gpt2")

prompts = [
    "The meaning of life is",
    "def fibonacci(n):",
    "Once upon a time",
    "The key to intelligence is",
    "import torch",
    "for i in range(10):",
]

for prompt in prompts:
    ids = tok.encode(prompt).ids[:10]
    gen = ids[:]
    for _ in range(100):
        inp = torch.tensor([gen[-SEQ:]], device=device)
        logits = model(inp)[0, -1, :]
        gen.append(sample(logits, temp=0.8, top_p=0.9).item())
    text = tok.decode(gen).replace("\u0120", " ").replace("\u010a", "\n")
    print(f"\n── {prompt} ──\n{text[:300]}")
