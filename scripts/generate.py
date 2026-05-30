"""独立生成测试: 加载最新 checkpoint, adaptive temperature + top-p + sampling."""
import sys, os, glob, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
from tokenizers import Tokenizer
from rina.mohe import MoHE
from rina.sample import sample

device = "cuda"
VOCAB, DM, SEQ = 50257, 1024, 64

ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../checkpoints/mohe_83m_resume.pt")
if not os.path.exists(ckpt_path):
    print("No checkpoint found at", ckpt_path)
    exit(1)

print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
sd = ckpt["model"]

model = MoHE(VOCAB, DM, 512, n_experts=4, aux_loss_weight=0.5, route_noise=0.0, expert_dropout=0.0, topk=2).to(device)
for k in list(sd.keys()):
    if "prev_route" in k or "_batch_" in k: del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()
torch.cuda.empty_cache()
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
    with torch.no_grad():
        for _ in range(100):
            inp = torch.tensor([gen[-SEQ:]], device=device)
            logits = model(inp)[0, -1, :]
            gen.append(sample(logits, temp=0.8, top_p=0.9).item())
    text = tok.decode(gen).replace("\u0120", " ").replace("\u010a", "\n")
    print(f"\n── {prompt} ──\n{text[:400]}")
torch.cuda.empty_cache()
