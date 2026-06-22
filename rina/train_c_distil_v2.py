"""Train L1 (Inertia Wave) with L2 (Route A) distillation"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_c import RINA_C, RINA_C_Config
from rina.model_a import RINA_A, RINA_A_Config

device = 'cuda'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def train(cfg, steps=100000, lr=1e-4, bsz=1, seq=512, out='models/out-0.1b-c-distil',
          resume=None, T=2.0, kl_alpha=0.5):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/mix_pretrain_llama.npy')
    os.makedirs(out, exist_ok=True)
    data = np.load(data_path, mmap_mode='r')
    N = len(data); val_len = N // 10

    def get_batch(s):
        pos = np.random.randint(0, N - val_len - s - 1, (bsz,))
        x = np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    # 学生：L1 Inertia Wave
    model = RINA_C(cfg).to(device)
    offset = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)['model']
        model.load_state_dict(ckpt, strict=False)
        base = int(os.path.splitext(os.path.basename(resume))[0].split('_')[-1]) if '_' in os.path.basename(resume).split('.')[0] and os.path.splitext(os.path.basename(resume))[0].split('_')[-1].isdigit() else 0
        offset = base
        print(f'Resumed from {resume} (step {base})')
    else:
        src = torch.load('models/out-0.1b-a-v2/a_final.pt', map_location=device, weights_only=False)['model']
        own = model.state_dict()
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v.to(device))
        model.load_state_dict(own, strict=False)
        print('Loaded shared weights from a_final.pt')
    model.train()

    # 老师：L2 Route A，冻结
    tea_cfg = RINA_A_Config(vocab_size=128256, block_size=seq, use_int4=True,
                             n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160)
    teacher = RINA_A(tea_cfg).to(device).half()
    tea_sd = torch.load('models/out-0.1b-a-v2/a_final.pt', map_location=device, weights_only=False)['model']
    teacher.load_state_dict(tea_sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print('Teacher L2 loaded (frozen, FP16)')

    # 还原 seq 到实际值（考试 FP16 后 rope buffer shape 必须匹配）
    teacher.config.block_size = seq
    for b in teacher.transformer.h:
        for m in [b.attn.rope, b.attn.rope_q]:
            cos = m.cos[:seq].float().half() if seq < m.cos.size(0) else m.cos.half()
            sin = m.sin[:seq].float().half() if seq < m.sin.size(0) else m.sin.half()
            m.cos = cos; m.sin = sin

    opt = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if p.dim()>=2], 'weight_decay': 0.01, 'initial_lr': lr},
        {'params': [p for n,p in model.named_parameters() if p.dim()<2], 'weight_decay': 0.0, 'initial_lr': lr},
    ], lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pbar = tqdm(range(steps)); t0 = time.time()
    log_file = open(f'{out}/train.log', 'a' if resume else 'w', buffering=1)
    if not resume:
        log_file.write('step\tce\tkl\ttotal\tval\tlr\n')

    for step in pbar:
        x = get_batch(seq + 1)
        inp, tgt = x[:, :-1], x[:, 1:]

        logits_s, loss_ce = model(inp, tgt)

        with torch.no_grad():
            logits_t, _, _ = teacher(inp, tgt)

        log_p_s = F.log_softmax(logits_s.float() / T, dim=-1)
        p_t = F.softmax(logits_t.float() / T, dim=-1)
        loss_kl = F.kl_div(log_p_s, p_t, reduction='batchmean') * T * T

        loss = loss_ce + kl_alpha * loss_kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                xv = get_batch(seq + 1)
                _, val_loss = model(xv[:, :-1], xv[:, 1:])
            model.train()
            msg = f'step {step+offset}: ce={loss_ce.item():.2f} kl={loss_kl.item():.2f} total={loss.item():.2f} val={val_loss.item():.2f} lr={sched.get_last_lr()[0]:.1e}'
            pbar.set_postfix_str(msg)
            log_file.write(msg + '\n')
            torch.save({'model': model.state_dict()}, f'{out}/c_{step+offset}.pt')

    torch.save({'model': model.state_dict()}, f'{out}/c_final.pt')
    log_file.close()
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100000); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--bsz', type=int, default=1); p.add_argument('--seq', type=int, default=512)
    p.add_argument('--out', type=str, default='models/out-0.1b-c-distil')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--T', type=float, default=2.0, help='蒸馏温度')
    p.add_argument('--kl_alpha', type=float, default=0.5, help='KL loss 权重')
    args = p.parse_args()
    cfg = RINA_C_Config(vocab_size=128256, block_size=args.seq, n_layer=16, n_embd=640, d_c=160, head_dim=64, n_head=10)
    train(cfg, steps=args.steps, lr=args.lr, bsz=args.bsz, seq=args.seq,
          out=args.out, resume=args.resume, T=args.T, kl_alpha=args.kl_alpha)
