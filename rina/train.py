"""RINA 训练入口：从零训标准 AR CE。
用法：python3 rina/train.py [--config 参数...]

数据：读取 checkpoints/mohe_fw_rwkv_1b.npy，纯英文数据请先运行 experiments/download_english.py
"""
import os, sys, time, math, argparse, numpy as np, torch, torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from .model import RINA, RINAConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(config: RINAConfig, **kwargs):
    lr = kwargs.get('lr', 3e-4)
    bsz = kwargs.get('bsz', 8)
    steps = kwargs.get('steps', 10000)
    out_dir = kwargs.get('out', 'checkpoints/rina')
    os.makedirs(out_dir, exist_ok=True)

    # ── 数据 ──
    data_paths = [
        'data/english_pretrain.npy',
        'checkpoints/mohe_fw_rwkv_1b.npy',
    ]
    data = None
    for p in data_paths:
        if os.path.exists(p):
            data = np.load(p, mmap_mode='r')
            print(f'Data: {p} — {len(data)/1e6:.0f}M tokens')
            break
    if data is None:
        raise FileNotFoundError('No training data found. Run experiments/download_english.py first.')

    N = len(data)
    val_len = N // 10
    val_rng = np.random.RandomState(42)

    def get_batch(s):
        pos = np.random.randint(0, N - val_len - s - 1, (bsz,))
        x = np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)
    def get_val(s):
        p = val_rng.randint(N - val_len, N - s - 1, (bsz,))
        x = np.array([data[p:p+s] for p in p])
        return torch.from_numpy(x).long().to(device)

    # ── 模型 ──
    model = RINA(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total/1e6:.2f}M')

    opt = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if p.dim()>=2], 'weight_decay': 0.01},
        {'params': [p for n,p in model.named_parameters() if p.dim()<2], 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    model.train()
    pbar = tqdm(range(steps))
    t0 = time.time()

    for step in pbar:
        x = get_batch(config.block_size)
        logits, loss = model(x, x)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                xv = get_val(config.block_size)
                lv, _ = model(xv, xv)
                ce_v = lv.mean().item()
            model.train()
            pbar.set_postfix(loss=f'{loss.item():.2f}', val=f'{ce_v:.2f}',
                             lr=f'{sched.get_last_lr()[0]:.1e}')
            torch.save({'model': model.state_dict(), 'step': step},
                       f'{out_dir}/rina_{step}.pt')

    torch.save({'model': model.state_dict()}, f'{out_dir}/rina_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=int, default=50257)
    p.add_argument('--dim', type=int, default=512)
    p.add_argument('--layers', type=int, default=12)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--kv-heads', type=int, default=4)
    p.add_argument('--d-c', type=int, default=128)
    p.add_argument('--seq', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--bsz', type=int, default=8)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--158', action='store_true', help='启用 1.58-bit 三元量化')
    p.add_argument('--int4', action='store_true', help='启用 int4/int2 量化感知训练')
    args = p.parse_args()

    cfg = RINAConfig(
        vocab_size=args.vocab, n_embd=args.dim, n_layer=args.layers,
        n_head=args.heads, n_kv_heads=args.kv_heads, d_c=args.d_c,
        block_size=args.seq, use_158=args.__dict__['158'], use_int4=args.int4,
    )
    train(cfg, lr=args.lr, bsz=args.bsz, steps=args.steps, out='checkpoints/rina')
