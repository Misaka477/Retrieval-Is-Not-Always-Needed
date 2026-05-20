"""
在线 Hebbian 自适应测试 — 推理时开启可塑性。

测:
  1. 训好的模型 (th=0.5, 无 Hebbian)
  2. 推理时开启 Hebbian (lr=0.001)
  3. ppl 随推理步数是否下降？
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from modules.temporal_snn_cell import TemporalSNNCell, TemporalSNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 配置 ──
DM = 128; NP = 512; V = 4096; SEQ = 128; BS = 8

# ── 加载 tokenizer ──
tok = Tokenizer.from_file(os.path.join(_ROOT, "checkpoints", "cann_15m_wt103-vocab.json"))
V_tok = tok.get_vocab_size()
print(f"Tokenizer: vocab={V_tok}")

# ── 合成测试数据 ──
import random
random.seed(99)
words = ["the","cat","sat","on","mat","dog","ran","in","park","a","is","was",
         "and","but","with","for","from","to","of","it","he","she","they"]
# 两个领域的数据: A (训练域), B (测试域 - 有不同词)
text_train = " ".join(random.choices(words[:12], k=15000))  # domain A
text_test  = " ".join(random.choices(words[8:], k=10000))  # domain B (shift)

ids_train = tok.encode(text_train).ids
ids_test  = tok.encode(text_test).ids

print(f"Train tokens: {len(ids_train)}  Test tokens: {len(ids_test)}")

# ── 训练一个基础模型 (th=0.5, 无 Hebbian) ──
EPOCHS = 5
train_t = torch.tensor(ids_train)
num_ba = (len(train_t) - 1) // (BS * SEQ)
print(f"Batches/epoch: {num_ba}")

model = TemporalSNNModel(V_tok, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=2, error_threshold=0.5,
                          hebbian_lr=0.0).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("\n── Phase 1: 离线训练 (无 Hebbian) ──")
for ep in range(EPOCHS):
    total_loss = 0.0
    for bi in range(num_ba):
        s = (bi * BS * SEQ) % (len(train_t) - BS * SEQ)
        x = train_t[s:s + BS * SEQ].view(BS, SEQ).to(device)
        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V_tok), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        total_loss += loss.item()
    ppl = torch.exp(torch.tensor(total_loss / num_ba)).item()
    print(f"  ep {ep+1}: ppl={ppl:.1f}  att={model.get_att_rate()*100:.0f}%")

# ── Phase 2: 在线推理 (测试域, 推理时强制 attractor 给 Hebbian 机会) ──
model_hebb = TemporalSNNModel(V_tok, d_model=DM, n_patterns=NP, beta=0.5,
                               attract_every=1, error_threshold=-1.0,
                               hebbian_lr=0.005).to(device)
model_hebb.cell.load_state_dict(model.cell.state_dict(), strict=False)
# force attractor ON during inference, but keep trained gate weights
model_hebb.cell.error_threshold[0] = -1.0
model_hebb.cell.attract_every = 1

test_t = torch.tensor(ids_test)
num_test_ba = (len(test_t) - 1) // (BS * SEQ)

def eval_online(m, batches, label):
    m.eval()
    ppls = []
    for bi in range(batches):
        s = (bi * BS * SEQ) % (len(test_t) - BS * SEQ)
        x = test_t[s:s + BS * SEQ].view(BS, SEQ).to(device)
        with torch.no_grad():
            logits = m(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V_tok), x[:, 1:].reshape(-1))
        ppls.append(torch.exp(loss).item())
    return ppls

print(f"\n── Phase 2: 在线推理 (测试域, {num_test_ba} batches) ──")

ppl_off = eval_online(model, num_test_ba, "Hebb OFF")
ppl_on  = eval_online(model_hebb, num_test_ba, "Hebb ON")

import statistics
early = slice(0, max(1, num_test_ba // 4))
late  = slice(max(1, 3 * num_test_ba // 4), num_test_ba)

print(f"\n── 结果 ──")
print(f"Hebb OFF:  early ppl={statistics.mean(ppl_off[early]):.1f}  "
      f"late ppl={statistics.mean(ppl_off[late]):.1f}")
print(f"Hebb ON:   early ppl={statistics.mean(ppl_on[early]):.1f}  "
      f"late ppl={statistics.mean(ppl_on[late]):.1f}")
print(f"Hebb OFF Δ:  {statistics.mean(ppl_off[late]) - statistics.mean(ppl_off[early]):+.1f}")
print(f"Hebb ON  Δ:  {statistics.mean(ppl_on[late]) - statistics.mean(ppl_on[early]):+.1f}")

if statistics.mean(ppl_on[late]) < statistics.mean(ppl_off[late]):
    print(f"\n  ✅ Hebbian improves ppl by {statistics.mean(ppl_off[late])-statistics.mean(ppl_on[late]):.1f}")
else:
    print(f"\n  ❌ Hebbian does NOT improve ppl")
