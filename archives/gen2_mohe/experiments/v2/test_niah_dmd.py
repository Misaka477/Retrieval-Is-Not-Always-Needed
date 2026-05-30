"""
NIAH 验证 — 线性 K (DMD/Koopman) 替换非线性 attractor 后的 recall 影响。

对比:
  - BASELINE: 原始非线性 softmax attractor
  - LINEAR:   线性 K 算子 (h_new = K @ h, 单次 matmul)
  - NONE:     无 attractor (h_new = h, identity baseline)
"""
import sys, os, time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


def load_ckpt(ckpt_path):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "cell.patterns" in st:
        patterns = st["cell.patterns"]
    elif "cell.U" in st and "cell.V" in st:
        patterns = st["cell.U"] @ st["cell.V"]
    else:
        raise KeyError(f"No patterns found. Keys: {list(st.keys())[:20]}")
    return patterns, st


def make_patterns_toy(dm, np, use_ckpt=None):
    if use_ckpt is not None:
        patterns_full, _ = load_ckpt(use_ckpt)
        patterns_full = patterns_full.to(device)
        if patterns_full.shape[1] > dm:
            U, S, Vt = torch.linalg.svd(patterns_full.float(), full_matrices=False)
            patterns = (U[:, :dm] * S[:dm]) @ Vt[:dm, :dm]
        else:
            patterns = patterns_full
        patterns = patterns[:min(np, patterns.shape[0])]
    else:
        patterns = torch.randn(np, dm, device=device) * 0.02
    return patterns


def make_niah(n, gap, n_keys=10):
    f = 2 * n_keys + 1
    x_list, y_list = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys + 1, (1,)).item()
        v = torch.randint(n_keys + 1, 2 * n_keys + 1, (1,)).item()
        x_list.append([k, v] + [f] * gap + [k])
        y_list.append(v)
    return torch.tensor(x_list), torch.tensor(y_list)


class LinearAttractorCell(nn.Module):
    """将 CANN cell 的 attractor 替换为线性 K 算子."""

    def __init__(self, d_model, K_matrix):
        super().__init__()
        self.d_model = d_model
        if K_matrix is not None:
            self.register_buffer("K", K_matrix)
        else:
            self.K = nn.Parameter(torch.eye(d_model) * 0.1)
        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_b = nn.Linear(d_model * 2, d_model)
        self.gate_alpha = nn.Linear(d_model * 2, d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, x, step=0):
        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(self.gate_a(combined))
        b = torch.sigmoid(self.gate_b(combined))
        xp = self.proj_in(x)
        h_ssm = a * h + b * xp
        alpha = torch.sigmoid(self.gate_alpha(combined))
        K = self.K if isinstance(self.K, nn.Parameter) else self.K
        attracted = h_ssm @ K.T
        h_new = h_ssm + alpha * (attracted - h_ssm)
        return self.norm(h_new)


class LinearAttractorModel(nn.Module):
    def __init__(self, vocab_size, d_model, K_matrix, n_slots):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cell = LinearAttractorCell(d_model, K_matrix)
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)
        self.slot_proj = nn.Linear(d_model, d_model)
        self.register_buffer("slot_table", torch.zeros(vocab_size, d_model))

    def slot_write(self, key_id, value_id):
        ve = self.embed(torch.tensor([value_id], device=self.slot_table.device))
        with torch.no_grad():
            self.slot_table[key_id] = self.slot_proj(ve).squeeze(0)

    def forward(self, x):
        bsz, seq_len = x.shape
        dm = self.d_model
        emb = self.embed(x)
        h = torch.zeros(bsz, dm, device=x.device)
        logits = []
        for t in range(seq_len - 1):
            h = self.cell(h, emb[:, t, :])
            logits.append(self.head(self.state_norm(h)))
        i_ext = self.slot_table[x[:, -1]]
        h = self.cell(h + i_ext, emb[:, -1, :])
        logits.append(self.head(self.state_norm(h)))
        return torch.stack(logits, dim=1)


class NoAttractorCell(nn.Module):
    """纯 SSM gate, 无 attractor."""
    def __init__(self, d_model):
        super().__init__()
        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_b = nn.Linear(d_model * 2, d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, x, step=0):
        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(self.gate_a(combined))
        b = torch.sigmoid(self.gate_b(combined))
        h_ssm = a * h + b * self.proj_in(x)
        return self.norm(h_ssm)


class NoAttractorModel(nn.Module):
    """纯 SSM gate, 无 attractor."""
    def __init__(self, vocab_size, d_model, n_slots):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cell = NoAttractorCell(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)
        self.slot_proj = nn.Linear(d_model, d_model)
        self.register_buffer("slot_table", torch.zeros(vocab_size, d_model))

    def slot_write(self, key_id, value_id):
        ve = self.embed(torch.tensor([value_id], device=self.slot_table.device))
        with torch.no_grad():
            self.slot_table[key_id] = self.slot_proj(ve).squeeze(0)

    def forward(self, x):
        bsz, seq_len = x.shape
        dm = self.d_model
        emb = self.embed(x)
        h = torch.zeros(bsz, dm, device=x.device)
        logits = []
        for t in range(seq_len - 1):
            h = self.cell(h, emb[:, t, :])
            logits.append(self.head(self.state_norm(h)))
        i_ext = self.slot_table[x[:, -1]]
        h = self.cell(h + i_ext, emb[:, -1, :])
        logits.append(self.head(self.state_norm(h)))
        return torch.stack(logits, dim=1)


class CANNModel(nn.Module):
    """标准 CANN-SSM cell (来自 v2 cell)."""
    def __init__(self, vocab_size, d_model, patterns, beta=0.5):
        super().__init__()
        import modules.cann_ssm as _c
        _c._setup_cuda_seq_v2 = lambda: False
        from modules.cann_ssm import CANNSSMCell
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cell = CANNSSMCell(d_model, n_patterns=patterns.shape[0],
                                beta=beta, attract_every=1)
        self.cell.patterns = nn.Parameter(patterns.clone())
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)
        self.slot_proj = nn.Linear(d_model, d_model)
        self.register_buffer("slot_table", torch.zeros(vocab_size, d_model))

    def slot_write(self, key_id, value_id):
        ve = self.embed(torch.tensor([value_id], device=self.slot_table.device))
        with torch.no_grad():
            self.slot_table[key_id] = self.slot_proj(ve).squeeze(0)

    def forward(self, x):
        bsz, seq_len = x.shape
        dm = self.d_model
        emb = self.embed(x)
        h = torch.zeros(bsz, dm, device=x.device)
        logits = []
        for t in range(seq_len - 1):
            h = self.cell(h, emb[:, t, :], step=t)
            logits.append(self.head(self.state_norm(h)))
        i_ext = self.slot_table[x[:, -1]]
        h = self.cell(h + i_ext, emb[:, -1, :], step=seq_len - 1)
        logits.append(self.head(self.state_norm(h)))
        return torch.stack(logits, dim=1)


def train_niah(model, name, gap, steps=80, mini_batch=32):
    train_x, train_y = make_niah(400, gap)
    test_x, test_y = make_niah(100, gap)
    model.to(device)
    model.slot_table.zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0.0

    for ep in range(steps):
        model.train()
        model.zero_grad()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), mini_batch):
            idx = perm[i:i+mini_batch]
            logits = model(train_x[idx].to(device))
            loss = F.cross_entropy(logits[:, -1], train_y[idx].to(device))
            loss.backward()
        opt.step()

        with torch.no_grad():
            for b in range(train_x.shape[0]):
                k, v = int(train_x[b, 0]), int(train_y[b])
                if k > 0 and v > 0:
                    model.slot_write(k, v)

        if ep % 10 == 9:
            model.eval()
            with torch.no_grad():
                lt = model(test_x.to(device))
            acc = (lt[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
            best = max(best, acc)
            print(f"  [{name}] gap={gap:3d} ep={ep+1:2d}: acc={acc*100:.0f}% best={best*100:.0f}%")

    return best


def main():
    DM = 128
    NP = 1024
    V = 22
    V_NIAH = V

    print(f"Model: dm={DM}, np={NP}, vocab={V}")
    torch.manual_seed(42)

    # 生成 toy patterns (模拟真实 pattern 的分布特性)
    patterns = torch.randn(NP, DM, device=device) * 0.02

    # 拟合线性 K
    print("Fitting linear K...")
    N = 4096
    torch.manual_seed(42)
    idx = torch.randint(0, NP, (N,), device=device)
    h_patt = patterns[idx] + torch.randn(N, DM, device=device) * 0.1
    h_patt = h_patt / h_patt.norm(dim=-1, keepdim=True)
    alpha_val = 0.1
    beta_val = 0.5
    scores = (h_patt @ patterns.T) * beta_val
    attn = torch.softmax(scores, dim=-1)
    y_patt = h_patt + alpha_val * (attn @ patterns - h_patt)
    K = torch.linalg.lstsq(h_patt, y_patt, rcond=None).solution.T
    print(f"  K shape: {K.shape}")

    results = {}
    for gap in [8, 16, 32]:
        print(f"\n── gap={gap} ──")

        model_baseline = CANNModel(V, DM, patterns, beta=beta_val)
        acc_baseline = train_niah(model_baseline, "BASELINE", gap)
        print(f"  BASELINE gap={gap}: best={acc_baseline*100:.0f}%")

        model_linear = LinearAttractorModel(V, DM, K, V_NIAH)
        acc_linear = train_niah(model_linear, "LINEAR", gap)
        print(f"  LINEAR   gap={gap}: best={acc_linear*100:.0f}%")

        model_none = NoAttractorModel(V, DM, V_NIAH)
        acc_none = train_niah(model_none, "NONE", gap)
        print(f"  NONE      gap={gap}: best={acc_none*100:.0f}%")

        results[gap] = (acc_baseline, acc_linear, acc_none)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Gap':>4} | {'BASELINE':>10} | {'LINEAR K':>10} | {'NONE':>10}")
    print("-" * 44)
    for gap in [8, 16, 32]:
        b, l, n = results[gap]
        print(f"{gap:4d} | {b*100:8.0f}%  | {l*100:8.0f}%  | {n*100:8.0f}%")


if __name__ == "__main__":
    main()
