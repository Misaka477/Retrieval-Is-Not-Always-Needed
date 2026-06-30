"""Train RINA-Jamba-QW3: QW2 + 表量化训练"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_jamba_qw3 import RINA_Jamba_QW3, QW3_Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(cfg, steps=50000, lr=1e-3, bsz=1, acc=2,     out='models/out-rina-jamba-qw3', resume=None):
    data_path = os.path.join(BASE, 'data/mix_pretrain_llama_v2.npy')
    os.makedirs(out, exist_ok=True)
    data = np.load(data_path, mmap_mode='r')
    N = len(data); val_len = N // 10
    log_file = open(os.path.join(out, 'train.log'), 'a' if resume else 'w')

    def get_batch(s):
        pos = np.random.randint(0, N - val_len - s - 1, (bsz,))
        x = np.array([data[p:p + s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model = RINA_Jamba_QW3(cfg).to(device)
    start_step = 0

    if resume:
        sd = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
        start_step = sd.get('step', 0)
        print(f'Resumed from step {start_step}')
    else:
        print('Loading from QW checkpoint...')
        sd = torch.load(os.path.join(BASE, 'models/out-rina-jamba-qw/jambaqw_final.pt'),
                        map_location='cpu', weights_only=False)
        model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)

    trainable = sum(p.numel() for p in model.parameters())
    print(f'Full training (weight_bits={cfg.weight_bits}): {trainable/1e6:.2f}M params')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pbar = tqdm(range(start_step, steps)); t0 = time.time()
    accum_step = 0

    for step in pbar:
        x = get_batch(cfg.block_size + 1)
        l, ce = model(x[:, :-1], x[:, 1:])
        ce = ce / acc
        ce.backward()
        accum_step += 1

        if accum_step % acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            opt.zero_grad()

        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                xv = get_batch(cfg.block_size + 1)
                lv, ce_v = model(xv[:, :-1], xv[:, 1:])
            model.train()
            msg = f'step={step} ce={ce.item() * acc:.2f} val={ce_v.item():.2f}'
            log_file.write(msg + '\n'); log_file.flush()
            pbar.set_postfix(ce=f'{ce.item() * acc:.2f}', val=f'{ce_v.item():.2f}')
        if step > 0 and step % 500 == 0:
            torch.save({'model': model.state_dict(), 'step': step}, f'{out}/jambaqw3_{step}.pt')
            keep = sorted([f for f in os.listdir(out) if f.startswith('jambaqw3_') and f.endswith('.pt')],
                          key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for f in keep[:-4]: os.remove(os.path.join(out, f))

    log_file.close()
    torch.save({'model': model.state_dict()}, f'{out}/jambaqw3_final.pt')
    print(f'Done in {(time.time() - t0) / 60:.1f}min')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bsz', type=int, default=1)
    p.add_argument('--acc', type=int, default=2)
    p.add_argument('--seq', type=int, default=512)
    p.add_argument('--out', type=str, default='models/out-rina-jamba-qw3')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--weight_bits', type=int, default=4)
    p.add_argument('--ssm_q', type=int, default=4)
    args = p.parse_args()
    cfg = QW3_Config(vocab_size=128256, block_size=args.seq, use_int4=True,
                     n_embd=640, n_layer=16, n_head=10, n_kv_heads=5,
                     d_c=160, head_dim=64,
                     sparse_k=16, sparse_window=32, sparse_local_w=4,
                     ssm_steps=3, quant_mode='q2k_q1v',
                     ssm_qbits=args.ssm_q, weight_bits=args.weight_bits, embed_quant=True)
    train(cfg, steps=args.steps, lr=args.lr, bsz=args.bsz, acc=args.acc, out=args.out, resume=args.resume)
