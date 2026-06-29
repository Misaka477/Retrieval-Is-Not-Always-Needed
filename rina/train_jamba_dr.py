"""Train RINA-Jamba-DR: 每层双路径 + per-token 动态路由
初始化：
  ssm_path   ← models/out-0.1b-c-distil-multistep/c_final.pt
  sparse_path ← models/out-0.1b-a-v2/a_final.pt
"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_jamba_dr import RINA_Jamba_DR, DR_Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(cfg, steps=50000, lr=1e-3, bsz=1, acc=2,
          out='models/out-rina-jamba-dr', resume=None):
    data_path = os.path.join(BASE, 'data/mix_pretrain_llama_v2.npy')
    os.makedirs(out, exist_ok=True)
    data = np.load(data_path, mmap_mode='r')
    N = len(data); val_len = N // 10
    log_file = open(os.path.join(out, 'train.log'), 'a' if resume else 'w')

    def get_batch(s):
        pos = np.random.randint(0, N - val_len - s - 1, (bsz,))
        x = np.array([data[p:p + s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model = RINA_Jamba_DR(cfg).to(device)

    if resume:
        sd = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
        start_step = sd.get('step', 0)
    else:
        def rename_ckpt(path, prefix_src, prefix_dst):
            sd = torch.load(path, map_location='cpu', weights_only=False)['model']
            renamed = {}
            for k, v in sd.items():
                if '.rope.' in k or 'inv_freq' in k or 'cos' in k or 'sin' in k:
                    continue
                k_new = k.replace(prefix_src, prefix_dst)
                renamed[k_new] = v
            return renamed

        print('Loading SSM weights from c_final.pt...')
        ssm_sd = rename_ckpt(
            os.path.join(BASE, 'models/out-0.1b-c-distil-multistep/c_final.pt'),
            '.attn.', '.ssm_path.')
        model.load_state_dict(ssm_sd, strict=False)
        print(f'  {len(ssm_sd)} SSM weights loaded')

        print('Loading Attention weights from a_final.pt...')
        attn_sd = rename_ckpt(
            os.path.join(BASE, 'models/out-0.1b-a-v2/a_final.pt'),
            '.attn.', '.sparse_path.')
        model.load_state_dict(attn_sd, strict=False)
        print(f'  {len(attn_sd)} Attention weights loaded')

        start_step = 0

    trainable = sum(p.numel() for p in model.parameters())
    print(f'Training: {trainable/1e6:.2f}M params (routing_penalty={cfg.routing_penalty})')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pbar = tqdm(range(start_step, steps)); t0 = time.time()
    accum_step = 0

    for step in pbar:
        x = get_batch(cfg.block_size + 1)
        l, loss, ce, alpha = model(x[:, :-1], x[:, 1:])
        (loss / acc).backward()
        accum_step += 1

        if accum_step % acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            opt.zero_grad()

        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                xv = get_batch(cfg.block_size + 1)
                lv, loss_v, ce_v, alpha_v = model(xv[:, :-1], xv[:, 1:])
            model.train()
            msg = f'step={step} ce={ce.item():.2f} val={ce_v.item():.2f} α={alpha.item():.3f}'
            log_file.write(msg + '\n'); log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.2f}', α=f'{alpha.item():.3f}')
        if step > 0 and step % 500 == 0:
            torch.save({'model': model.state_dict(), 'step': step}, f'{out}/jambadr_{step}.pt')
            keep = sorted([f for f in os.listdir(out)
                          if f.startswith('jambadr_') and f.endswith('.pt')],
                          key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for f in keep[:-4]: os.remove(os.path.join(out, f))

    log_file.close()
    torch.save({'model': model.state_dict()}, f'{out}/jambadr_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bsz', type=int, default=1)
    p.add_argument('--acc', type=int, default=2)
    p.add_argument('--seq', type=int, default=512)
    p.add_argument('--out', type=str, default='models/out-rina-jamba-dr')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--weight_bits', type=int, default=4)
    p.add_argument('--ssm_q', type=int, default=4)
    p.add_argument('--routing_penalty', type=float, default=0.1)
    args = p.parse_args()
    cfg = DR_Config(vocab_size=128256, block_size=args.seq, use_int4=True,
                     n_embd=640, n_layer=16, n_head=10, n_kv_heads=5,
                     d_c=160, head_dim=64,
                     sparse_k=16, sparse_window=32, sparse_local_w=4,
                     ssm_steps=3, quant_mode='q2k_q1v',
                     ssm_qbits=args.ssm_q, weight_bits=args.weight_bits,
                     routing_penalty=args.routing_penalty)
    train(cfg, steps=args.steps, lr=args.lr, bsz=args.bsz, acc=args.acc,
          out=args.out, resume=args.resume)
