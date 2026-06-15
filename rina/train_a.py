"""Train Route A: full attention + latent contrastive loss"""
import os, sys, math, time, argparse
import torch, numpy as np
from tqdm import tqdm
from rina.model_a import RINA_A, RINA_A_Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(cfg, data_path='data/english_pretrain.npy', lr=3e-4, bsz=4, steps=10000, out='checkpoints/rina_a'):
    os.makedirs(out, exist_ok=True)
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_path)
    data = np.load(data_path, mmap_mode='r')
    N = len(data); val_len = N // 10

    def get_batch(s):
        pos = np.random.randint(0, N - val_len - s - 1, (bsz,))
        x = np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model = RINA_A(cfg).to(device); model.train()
    opt = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if p.dim()>=2], 'weight_decay': 0.01},
        {'params': [p for n,p in model.named_parameters() if p.dim()<2], 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pbar = tqdm(range(steps)); t0 = time.time()

    for step in pbar:
        x = get_batch(cfg.block_size + 1)
        l, ce_loss, lats = model(x[:, :-1], x[:, 1:])
        cl_loss = model.compute_contrastive_loss(lats, x[:, 1:]) * 3.0
        loss = ce_loss + cl_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                xv = get_batch(cfg.block_size + 1)
                lv, val_loss, _ = model(xv[:, :-1], xv[:, 1:])
                ce_v = val_loss.item()
            model.train()
            pbar.set_postfix(loss=f'{ce_loss.item():.2f}', val=f'{ce_v:.2f}', cl=f'{cl_loss.item():.2f}', lr=f'{sched.get_last_lr()[0]:.1e}')
            torch.save({'model': model.state_dict(), 'step': step}, f'{out}/rina_a_{step}.pt')

    torch.save({'model': model.state_dict()}, f'{out}/rina_a_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=10000); p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--bsz', type=int, default=4); p.add_argument('--seq', type=int, default=512)
    p.add_argument('--data', type=str, default='data/english_pretrain.npy')
    p.add_argument('--out', type=str, default='nanoGPT/out-rina-a-v2')
    p.add_argument('--dim', type=int, default=512); p.add_argument('--layers', type=int, default=12)
    p.add_argument('--heads', type=int, default=8); p.add_argument('--kv-heads', type=int, default=4)
    p.add_argument('--d-c', type=int, default=128)
    p.add_argument('--int4', action='store_true')
    args = p.parse_args()
    cfg = RINA_A_Config(vocab_size=50257, block_size=args.seq, n_layer=args.layers,
                        n_head=args.heads, n_kv_heads=args.kv_heads, n_embd=args.dim, d_c=args.d_c,
                        use_int4=args.int4)
    train(cfg, data_path=args.data, lr=args.lr, bsz=args.bsz, steps=args.steps, out=args.out)
