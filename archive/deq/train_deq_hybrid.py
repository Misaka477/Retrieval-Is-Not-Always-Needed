"""4-way comparison: TemporalSNN (Hebb+Inhib / Hebb only / No Hebb) vs DEQ-Hybrid.
DEQ-Hybrid: attractor uses detach(P) → gradients don't chase Hebbian-modified patterns.
dm=256, np=1024, WikiText-103 subset, ~1M tokens, 5 epoch, subsample=2."""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

import torch, torch.nn.functional as F, time
from tqdm import tqdm
from modules.temporal_snn_cell import TemporalSNNModel, TemporalSNNCell


# ══════ DEQ-Hybrid Cell ══════
class DEQHybridCell(TemporalSNNCell):
    """attractor 用 P.detach() — Hebbian 缓存到序列末统一更新，避开 inplace 版本冲突."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hebbian_buffer = []  # (pk, lr, dh) 元组

    def flush_hebbian(self):
        """序列 forward 完成后统一应用 Hebbian (解决 inplace 版本冲突)"""
        if not self._hebbian_buffer:
            return
        with torch.no_grad():
            for pk, lr, dh in self._hebbian_buffer:
                self.patterns.data.index_add_(0, pk, lr.unsqueeze(-1) * dh)
                for upk in pk.unique().tolist():
                    self.patterns.data[upk] *= self.hebbian_decay
        self.hebbian_updates += len(self._hebbian_buffer)
        self._hebbian_buffer.clear()
    def forward(self, h, x, step=0):
        bsz, dm = h.shape
        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(self.gate_a(combined))
        b = torch.sigmoid(self.gate_b(combined))
        h_ssm = a * h + b * self.proj_in(x)

        error = (h_ssm - h.detach()).norm(dim=-1) / (h.detach().norm(dim=-1) + 1e-8)
        is_att_step = (step % self.attract_every == (self.attract_every - 1))
        need_att = torch.ones(bsz, dtype=torch.bool, device=h.device) if self.error_threshold[0] < 0 else (error > self.error_threshold[0])
        do_att = is_att_step & need_att

        if self.training:
            self.total_steps += bsz
            self.att_calls += do_att.float().sum().detach()

        if do_att.any():
            P_stable = self.effective_patterns if not hasattr(self, 'patterns') or self.patterns is None else self.patterns
            P_stable = P_stable.detach()
            pat = P_stable.unsqueeze(0).expand(bsz, -1, -1)
            scores = (h_ssm.unsqueeze(1) @ pat.transpose(1, 2)) * self.beta_t[0]
            attn = torch.softmax(scores, dim=-1)
            attracted = (attn @ pat).squeeze(1)
            h_attracted = h_ssm + torch.sigmoid(self.gate_alpha(combined)) * (attracted - h_ssm)

            with torch.no_grad():
                if self.training:
                    # Hebbian 缓存到序列末 — 避开 inplace 版本冲突
                    k_pred = scores.argmax(dim=-1).squeeze(-1)
                    lr = self.hebbian_lr / (1.0 + error); lr = lr.clamp(max=self.hebbian_lr)
                    active = do_att.nonzero(as_tuple=True)[0]
                    if len(active) > 0:
                        pk = k_pred[active]; dh = h_attracted[active] - self.patterns[pk]
                        self._hebbian_buffer.append((pk.detach(), lr[active].detach(), dh.detach()))
                    self.hebbian_updates += do_att.float().sum().detach()

            mask = do_att.float().unsqueeze(-1)
            h_new = mask * h_attracted + (1.0 - mask) * h_ssm
        else:
            h_new = h_ssm

        return self.norm(h_new)


class DEQHybridModel(torch.nn.Module):
    def __init__(self, vocab_size, d_model=256, n_patterns=256, beta=0.5,
                 attract_every=2, error_threshold=0.5, hebbian_lr=0.01,
                 inhibition_threshold=0.0):
        super().__init__()
        self.d_model, self.attract_every = d_model, attract_every
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.cell = DEQHybridCell(d_model, n_patterns=n_patterns, beta=beta,
                                   attract_every=attract_every,
                                   error_threshold=error_threshold,
                                   hebbian_lr=hebbian_lr,
                                   inhibition_threshold=inhibition_threshold)
        self.head = torch.nn.Linear(d_model, vocab_size)
        self.state_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        bsz, sl = x.shape; dm = self.d_model
        emb = self.embed(x); h = torch.zeros(bsz, dm, device=x.device)
        logits = []
        for t in range(sl):
            if t < sl - 1:
                h = self.cell(h, emb[:, t, :], step=t)
            else:
                h = self.cell(h, emb[:, t, :], step=t)
                pat = self.cell.patterns.unsqueeze(0).expand(bsz, -1, -1)
                attracted = torch.softmax((h.unsqueeze(1) @ pat.transpose(1, 2)) * self.cell.beta_t[0], dim=-1) @ pat
                attracted = attracted.squeeze(1)
                combined_last = torch.cat([h, emb[:, -1, :]], dim=-1)
                h = h + torch.sigmoid(self.cell.gate_alpha(combined_last)) * (attracted - h)
                h = self.cell.norm(h)
            logits.append(self.head(self.state_norm(h)))
        self.cell.flush_hebbian()
        return torch.stack(logits, dim=1)

    def get_att_rate(self):
        return self.cell.att_rate


# ══════ Data ══════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"Device: {device}", flush=True)

print("Loading...", flush=True)
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
raw = [t for t in ds["text"] if len(t) > 100][:20000]
print(f"  segments: {len(raw)}", flush=True)

print("Training BPE...", flush=True)
tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trn = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<pad>"])
tok.train_from_iterator(raw[:8000], trn); tok.add_special_tokens(["<pad>"])
V = tok.get_vocab_size()
print(f"  vocab: {V}", flush=True)

print("Tokenizing...", flush=True)
il = [torch.tensor(tok.encode(t).ids, dtype=torch.long) for t in raw if len(tok.encode(t).ids) > 64]
ids = torch.cat(il)[:1000000]
print(f"  tokens: {len(ids):,}", flush=True)

# ══════ Config ══════
DM, NP, SEQ, BS, AE = 256, 1024, 64, 8, 2
EPOCHS, LR, SS = 5, 3e-4, 2
TH, HL, INH = 0.5, 0.01, 0.8
nt = (len(ids) - 1) // (BS * SEQ) // SS
print(f"  batches/epoch: {nt}", flush=True)

# ══════ Training ══════
def train_one(name, use_deq=False, use_hebb=True, use_inhib=True):
    hb_lr = HL if use_hebb else 0.0
    inh   = INH if (use_hebb and use_inhib) else 0.0
    if use_deq:
        m = DEQHybridModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                            attract_every=AE, error_threshold=TH,
                            hebbian_lr=hb_lr, inhibition_threshold=inh).to(device)
    else:
        m = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                              attract_every=AE, error_threshold=TH,
                              hebbian_lr=hb_lr, inhibition_threshold=inh).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        m.train(); tl = 0.0
        perm = torch.randperm(len(ids) - BS * SEQ)
        pbar = tqdm(range(nt), desc=f"[{name}] ep{ep}/{EPOCHS}", leave=False)
        for bi in pbar:
            s = perm[(bi * SS) % len(perm)]
            x = ids[s:s + BS * SEQ].view(BS, SEQ).to(device)
            opt.zero_grad()
            logits = m(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); tl += loss.item()
            if bi % 100 == 99:
                ppl = torch.exp(torch.tensor(tl / (bi + 1))).item()
                pbar.set_postfix(loss=f"{tl/(bi+1):.3f}", ppl=f"{ppl:.1f}")
        ppl = torch.exp(torch.tensor(tl / nt)).item()
        att = m.get_att_rate()
        print(f"  [{name}] ep{ep}: ppl={ppl:.1f} att={att*100:.0f}%", flush=True)
    return ppl, time.time() - t0


print(f"\n{'='*60}")
print(f"4-way: dm={DM} np={NP} ep={EPOCHS} tokens={len(ids):,}")
print(f"{'='*60}")

configs = [
    # ("+Hebb+Inhib  ", False, True,  True),
    # ("+Hebb only   ", False, True,  False),
    # ("No Hebb       ", False, False, False),
    ("DEQ-Hybrid   ", True,  True,  True),
]

results = {}
for name, deq, hb, inh in configs:
    print(f"\n-- {name} --", flush=True)
    ppl, elapsed = train_one(name, use_deq=deq, use_hebb=hb, use_inhib=inh)
    results[name] = (ppl, elapsed)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"{'Method':>16} {'PPL':>8} {'Time':>8}")
print("-" * 36)
for name in [c[0] for c in configs]:
    p, t = results[name]
    print(f"{name:>16} {p:8.1f} {t/60:7.1f}m")
