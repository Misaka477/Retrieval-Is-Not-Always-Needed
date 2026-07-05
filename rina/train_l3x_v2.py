"""Train RINA L3X V2 — SSM-primary hybrid with cluster-compressed sparse attention"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_l3x_v2 import RINA_L3X_V2, L3XV2_Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train(cfg, steps=50000, lr=1e-3, bsz=2, seq=512,
          out='models/out-l3x-v2', resume=None, warmup=2000,
          cl_weight=0.0, grad_accum=2, ckpt_limit=5, val_every=5000):
    data_path = os.path.join(BASE, 'data/dclm_pretrain_llama_2000m.npy')
    os.makedirs(out, exist_ok=True)
    data = np.load(data_path, mmap_mode='r')
    N = len(data)
    val_len = N // 50
    log_file = open(os.path.join(out, 'train.log'), 'a' if resume else 'w', buffering=1)
    if not resume:
        log_file.write('step\tce\tcl\tval\tlr\tssm_depth\n')

    def get_batch(s):
        pos = np.random.randint(0, N - val_len - s - 1, (bsz,))
        x = np.array([data[p:p + s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model = RINA_L3X_V2(cfg).to(device)
    start_step = 0
    if resume:
        sd = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
        start_step = sd.get('step', 0)
        print(f'Resumed from step {start_step}')

    trainable = sum(p.numel() for p in model.parameters())
    eff_bsz = bsz * grad_accum
    print(f'Trainable: {trainable/1e6:.1f}M params | bsz={bsz} ga={grad_accum} eff={eff_bsz} cl={cl_weight}')

    opt = torch.optim.AdamW(
        [{'params': [p for n,p in model.named_parameters() if p.dim()>=2],
          'weight_decay': 0.1},
         {'params': [p for n,p in model.named_parameters() if p.dim()<2],
          'weight_decay': 0.0}],
        lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, steps, eta_min=lr*0.01)
    pbar = tqdm(range(start_step, steps))
    t0 = time.time()
    best_val = float('inf')

    for step in pbar:
        if step < warmup:
            for g in opt.param_groups:
                g['lr'] = lr * (step + 1) / warmup

        loss_acc = 0
        cl = None
        for micro in range(grad_accum):
            x = get_batch(seq + 1)
            if cl_weight > 0:
                l, ce, lats, ssm_depth = model(x[:, :-1], x[:, 1:], return_lats=True)
                cl = model.compute_contrastive_loss(lats) * cl_weight
                micro_loss = (ce + cl) / grad_accum
            else:
                l, ce, ssm_depth = model(x[:, :-1], x[:, 1:])
                micro_loss = ce / grad_accum
            micro_loss.backward()
            loss_acc += micro_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if step >= warmup:
            sched.step()

        if step % 100 == 0 and step > 0:
            lr_cur = opt.param_groups[0]['lr']
            cl_s = f' cl={cl.item():.4f}' if cl is not None else ''
            pbar.set_description(
                f'ce={loss_acc:.3f}{cl_s} sd={ssm_depth:.3f} lr={lr_cur:.2e}')
            cl_t = f'\t{cl.item():.4f}' if cl is not None else '\t-'
            log_file.write(
                f'{step}\t{loss_acc:.4f}{cl_t}\t-\t{lr_cur:.2e}\t{ssm_depth:.4f}\n')
            log_file.flush()

        if val_every > 0 and step % val_every == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                xv = get_batch(seq + 1)
                _, val_ce, _ = model(xv[:, :-1], xv[:, 1:])
            model.train()
            cl_s = f'\t{cl.item():.4f}' if cl is not None else '\t-'
            log_file.write(
                f'{step}\t{loss_acc:.4f}{cl_s}\t{val_ce.item():.4f}\t'
                f'{opt.param_groups[0]["lr"]:.2e}\t{ssm_depth:.4f}\n')
            log_file.flush()

            if val_ce.item() < best_val:
                best_val = val_ce.item()
                torch.save({
                    'model': model.state_dict(),
                    'step': step,
                    'ce': loss_acc,
                    'val_ce': val_ce.item(),
                }, os.path.join(out, 'best.pt'))

        if step > 0 and step % 5000 == 0:
            torch.save({
                'model': model.state_dict(),
                'step': step,
            }, os.path.join(out, f'step_{step}.pt'))
            # auto-rotate: keep only latest ckpt_limit step checkpoints
            existing=sorted([f for f in os.listdir(out) if f.startswith('step_') and f.endswith('.pt')], reverse=True)
            for stale in existing[ckpt_limit:]:
                os.remove(os.path.join(out, stale))

    log_file.close()
    torch.save({'model': model.state_dict(), 'step': steps},
               os.path.join(out, 'final.pt'))
    elapsed = time.time() - t0
    print(f'Done in {elapsed/3600:.1f}h. Best val CE: {best_val:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--seq', type=int, default=2048)
    parser.add_argument('--out', type=str, default='models/out-l3x-v2')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--cl-weight', type=float, default=1.0)
    parser.add_argument('--no-cl', action='store_true')
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--ckpt-limit', type=int, default=5)
    parser.add_argument('--val-every', type=int, default=5000)
    args = parser.parse_args()

    seq=min(args.seq, 2048)
    cfg=L3XV2_Config(block_size=seq)
    train(cfg, steps=args.steps, lr=args.lr, bsz=args.bsz, seq=seq,
          out=args.out, resume=args.resume,
          cl_weight=0.0 if args.no_cl else args.cl_weight,
          grad_accum=args.grad_accum, ckpt_limit=args.ckpt_limit,
          val_every=args.val_every)
