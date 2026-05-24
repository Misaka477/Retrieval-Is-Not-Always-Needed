"""诊断 MoHE ep1: 路由行为 + 专家分化 + pattern 分析."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rina.mohe import MoHE
device = "cuda"
VOCAB, DM = 50257, 256
ckpt_dir = "checkpoints"
ckpt_path = os.path.join(ckpt_dir, "mohe_large_ep1.pt")
if not os.path.exists(ckpt_path):
    print("No mohe_large checkpoint found"); exit(1)
print(f"Loading {os.path.basename(ckpt_path)}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = MoHE(VOCAB, DM, 512, n_experts=4).to(device)
sd = ckpt["model"]
for k in list(sd.keys()):
    if k.startswith("prev_route"): del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()

print("\n=== Expert patterns 相似度 ===")
for i in range(4):
    for j in range(i+1, 4):
        pi = model.experts[i].patterns.data
        pj = model.experts[j].patterns.data
        sim = (pi @ pj.T).mean().item()
        cd = (pi @ pj.T).clamp(-1,1).mean().item()
        print(f"  {i} vs {j}: sim={sim:.4f} cos={cd:.4f}")

print("\n=== Router 权重 ===")
rw = model.router.weight.data
print(f"  shape={rw.shape} mean={rw.mean():.4f} std={rw.std():.4f}")

print("\n=== 路由采样 ===")
from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("gpt2")
prompts = ["def fibonacci(n):", "The meaning of life is",
           "import torch", "for i in range(10):", "x = [1, 2, 3]"]
for prompt in prompts:
    ids = tok.encode(prompt).ids[:16]
    inp = torch.tensor([ids], device=device)
    emb = model.embed_norm(model.embed(inp))
    h = torch.zeros(1, DM, device=device)
    routes = []
    for t in range(min(len(ids), 16)):
        x_emb = emb[:, t, :]
        combined = torch.cat([h, x_emb], dim=-1)
        raw = model.router(combined)
        w = torch.softmax(raw, dim=-1)[0].detach().cpu()
        routes.append(w)
        h_outs = [exp(h, x_emb)[0] for exp in model.experts]
        h = model.consolidate_norm(model.consolidate(torch.cat(h_outs, dim=-1)))
    r = torch.stack(routes)
    print(f"  [{prompt[:20]:20s}] avg={r.mean(0).numpy().round(3)} std={r.std(0).numpy().round(3)}")

print("\n=== Gate_a 权重差异 ===")
for i in range(4):
    for j in range(i+1, 4):
        d = (model.experts[i].gate_a.weight - model.experts[j].gate_a.weight).norm().item()
        print(f"  {i} vs {j}: {d:.4f}")
