"""NIAH with real text: train Hopfield + Slot on needle recall from P&P context."""
import torch, torch.nn.functional as F, sys, math, time
sys.path.insert(0, "D:\\Software_Development\\Project\\RINA_Project\\references\\hopfield-layers")
from hflayers import Hopfield
device = "cuda"

# ── 100-char vocab ──
import string
chars = string.printable.strip()
stoi = {c: i+3 for i, c in enumerate(chars)}
stoi["<pad>"] = 0; stoi["<bos>"] = 1; stoi["<eos>"] = 2
itos = {i: c for c, i in stoi.items()}
V = len(stoi)

# ── Load P&P ──
pp = open("D:\\Software_Development\\Project\\RINA_Core\\scripts\\evaluation\\_pride.txt",
          "r", encoding="latin-1").read()
ci = pp.find("CHAPTER")
pp = pp[ci:ci+50000] if ci >= 0 else pp[:50000]
pp_ids = torch.tensor([stoi.get(c, 0) for c in pp])

needle_str = "KILO42"
q_str = "password"
nd = torch.tensor([stoi.get(c, 0) for c in needle_str])
qu = torch.tensor([stoi.get(c, 0) for c in q_str])

# ── Model: Hopfield + GPU slot ──
class HopfieldSlotLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dm = 128
        self.embed = torch.nn.Embedding(V, dm)
        self.hopfield = Hopfield(dm, dm, dm, num_heads=1, scaling=0.5,
                                 update_steps_max=3, batch_first=True)
        self.norm = torch.nn.LayerNorm(dm)
        self.head = torch.nn.Linear(dm, V)
        self.slot = torch.zeros(V, dtype=torch.long, device=device)  # key -> value_token
    def forward(self, x):
        out = self.head(self.norm(self.hopfield(self.embed(x))))
        # Inject slot at last position
        tid = x[0, -1].item()
        sv = self.slot[tid].item()
        if sv > 0:
            out[:, -1, sv] += 5.0  # strong bias toward slot value
        return out

m = HopfieldSlotLM().to(device)
print(f"Params: {sum(p.numel() for p in m.parameters()):,}")

# ── Train with slot writes ──
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
t0 = time.time()
for ep in range(60):
    m.train()
    tl = 0; nb = 0
    for _ in range(150):
        start = torch.randint(0, len(pp_ids) - 200, (16,))
        x = torch.stack([pp_ids[s:s+128] for s in start]).to(device)
        y = torch.stack([pp_ids[s+1:s+129] for s in start]).to(device)
        # Insert needle + query at end of each sequence
        for b in range(16):
            needle_pos = 96
            x[b, needle_pos:needle_pos+len(nd)] = nd.to(device)
            x[b, needle_pos+len(nd):needle_pos+len(nd)+len(qu)] = qu.to(device)
            # Target: needle follows query
            y[b, needle_pos+len(nd)-1:needle_pos+len(nd)-1+len(nd)] = nd.to(device)

        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item(); nb += 1

    # Slot writes from prediction errors
    with torch.no_grad():
        for _ in range(20):
            start = torch.randint(0, len(pp_ids) - 200, (8,))
            xb = torch.stack([pp_ids[s:s+128] for s in start]).to(device)
            yb = torch.stack([pp_ids[s+1:s+129] for s in start]).to(device)
            for b in range(8):
                xb[b, 96:96+len(nd)] = nd.to(device)
                xb[b, 96+len(nd):96+len(nd)+len(qu)] = qu.to(device)
                yb[b, 95:95+len(nd)] = nd.to(device)
            logits = m(xb)
            for b in range(8):
                for t in range(127):
                    if t >= 96+len(nd)-1 and t < 96+len(nd)-1+len(nd):
                        # Check if model correctly predicted needle char
                        pred = logits[b, t].argmax().item()
                        actual = yb[b, t].item()
                        if pred != actual:
                            m.slot[xb[b, t].item()] = actual

    if ep % 10 == 9:
        active = (m.slot > 0).sum().item()
        print(f"ep={ep}: loss={tl/nb:.4f} slot_active={active} ({time.time()-t0:.0f}s)")

# ── Generate: autoregressive with slot ──
m.eval()
prompt = "password "
x = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
print(f"\nPrompt: {repr(prompt)}")
print("Generated: ", end="", flush=True)
with torch.no_grad():
    for _ in range(20):
        logits = m(x)
        # Slot injection at last position
        tid = x[0, -1].item()
        sv = m.slot[tid].item()
        if sv > 0:
            logits[:, -1, sv] += 10.0  # very strong bias
        nid = logits[0, -1].argmax().item()
        ch = itos.get(nid, "?")
        print(ch, end="", flush=True)
        x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
print()

# ── NIAH eval on longer contexts ──
print("\n=== NIAH Evaluation ===")
for ctx_len in [256, 512]:
    for depth in [0.25, 0.5, 0.75]:
        needle_pos = int(ctx_len * depth)
        ctx = pp_ids[max(0, needle_pos-50):needle_pos+ctx_len].clone()
        # Insert needle at needle_pos
        ins_pos = min(50, len(ctx) - len(nd) - len(qu) - 5)
        ctx[ins_pos:ins_pos+len(nd)] = nd
        ctx[ins_pos+len(nd):ins_pos+len(nd)+len(qu)] = qu
        x = ctx[:ctx_len+len(qu)].unsqueeze(0).to(device)

        gen = ""
        with torch.no_grad():
            for _ in range(30):
                logits = m(x)
                tid = x[0, -1].item()
                sv = m.slot[tid].item()
                if sv > 0:
                    logits[:, -1, sv] += 10.0
                nid = logits[0, -1].argmax().item()
                gen += itos.get(nid, "?")
                x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)

        passed = "KILO42" in gen
        print(f"  ctx={ctx_len:3d} depth={depth:.2f}: {'PASS' if passed else 'FAIL'} gen={repr(gen[:30])}")
