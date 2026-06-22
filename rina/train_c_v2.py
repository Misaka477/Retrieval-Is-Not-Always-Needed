"""Train Route C 0.1B: Inertia Wave，CF 架构的 L1 路由"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_c import RINA_C, RINA_C_Config

device = 'cuda'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def train(cfg, steps=100000, lr=1e-4, bsz=2, seq=512, out='models/out-0.1b-c', resume=None):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/mix_pretrain_llama.npy')
    os.makedirs(out, exist_ok=True)
    data = np.load(data_path, mmap_mode='r')
    N = len(data); val_len = N // 10

    def get_batch(s):
        pos = np.random.randint(0, N - val_len - s - 1, (bsz,))
        x = np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model = RINA_C(cfg).to(device)
    offset = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)['model']
        model.load_state_dict(ckpt, strict=False)
        base = int(os.path.splitext(os.path.basename(resume))[0].split('_')[-1]) if '_' in os.path.basename(resume).split('.')[0] else 0
        offset = base
        print(f'Resumed from {resume} (step {base}), remaining={steps} steps')
    else:
        src = torch.load('models/out-0.1b-a-v2/a_final.pt', map_location=device, weights_only=False)['model']
        own = model.state_dict()
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v.to(device))
        model.load_state_dict(own, strict=False)
        print('Loaded shared weights from Route A final')
    model.train()

    opt = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if p.dim()>=2], 'weight_decay': 0.01, 'initial_lr': lr},
        {'params': [p for n,p in model.named_parameters() if p.dim()<2], 'weight_decay': 0.0, 'initial_lr': lr},
    ], lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pbar = tqdm(range(steps))
    log_file = open(f'{out}/train.log', 'a' if resume else 'w', buffering=1)
    if not resume:
        log_file.write('step\tce\tval\tlr\n')

    for step in pbar:
        x = get_batch(seq + 1)
        logits, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                xv = get_batch(seq + 1)
                _, val_loss = model(xv[:, :-1], xv[:, 1:])
            model.train()
            msg = f'step {step+offset}: ce={loss.item():.2f} val={val_loss.item():.2f} lr={sched.get_last_lr()[0]:.1e}'
            pbar.set_postfix_str(msg)
            log_file.write(msg + '\n')
            torch.save({'model': model.state_dict()}, f'{out}/c_{step+offset}.pt')

    torch.save({'model': model.state_dict()}, f'{out}/c_final.pt')
    log_file.close()
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100000); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--bsz', type=int, default=2); p.add_argument('--seq', type=int, default=512)
    p.add_argument('--out', type=str, default='models/out-0.1b-c')
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()
    cfg = RINA_C_Config(vocab_size=128256, block_size=args.seq, n_layer=16, n_embd=640, d_c=160, head_dim=64, n_head=10)
    train(cfg, steps=args.steps, lr=args.lr, bsz=args.bsz, seq=args.seq, out=args.out, resume=args.resume)
