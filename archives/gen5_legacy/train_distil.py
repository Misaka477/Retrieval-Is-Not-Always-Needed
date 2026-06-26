"""Distil 0.1B: 用 Llama 3.2 1B 做 teacher，蒸馏训练"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm

device = 'cuda'

def train(cfg, total_steps=100000, lr=3e-4, bsz=4, seq=1024, out='models/out-0.1b-distil'):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/mix_pretrain_llama.npy')
    os.makedirs(out, exist_ok=True)
    data = np.load(data_path, mmap_mode='r') if os.path.exists(data_path) else None
    
    if data is None:
        print('等待数据重 tokenize 完成...')
        return
    
    N = len(data); val_len = N // 10

    def get_batch(s):
        pos = np.random.randint(0, N - val_len - s - 1, (bsz,))
        x = np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    # 学生: RINA_0.1B with Llama vocab
    from rina.model_a import RINA_A, RINA_A_Config
    cfg_stu = RINA_A_Config(vocab_size=128256, block_size=seq, use_int4=True,
                            n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160)
    stu = RINA_A(cfg_stu).to(device); stu.train()

    # 老师: Llama 3.2 1B
    from transformers import AutoModelForCausalLM
    tea = AutoModelForCausalLM.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct',
                                               torch_dtype=torch.float16, device_map='cuda')
    tea.eval()
    for p in tea.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW([
        {'params': [p for n,p in stu.named_parameters() if p.dim()>=2], 'weight_decay': 0.01},
        {'params': [p for n,p in stu.named_parameters() if p.dim()<2], 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    pbar = tqdm(range(total_steps)); t0 = time.time()

    for step in pbar:
        x = get_batch(seq + 1)
        x_in, x_tgt = x[:,:-1], x[:,1:]

        # 学生 forward
        l_stu, _, _ = stu(x_in, x_tgt)

        # 老师 forward（蒸馏目标）
        with torch.no_grad():
            try:
                l_tea = tea(x_in).logits  # [B, T, V]
            except:
                l_tea = tea(x_in, labels=x_tgt).logits

        # CE loss
        ce = F.cross_entropy(l_stu.view(-1, l_stu.size(-1)), x_tgt.reshape(-1), ignore_index=-1)

        # KL 蒸馏 loss
        T = 2.0  # 蒸馏温度
        p_stu = F.log_softmax(l_stu / T, dim=-1)
        p_tea = F.softmax(l_tea / T, dim=-1)
        kl = F.kl_div(p_stu.view(-1, p_stu.size(-1)), p_tea.view(-1, p_tea.size(-1)),
                      reduction='batchmean') * (T ** 2)

        loss = ce + 0.5 * kl

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 500 == 0:
            stu.eval()
            with torch.no_grad():
                xv = get_batch(seq + 1)
                lv, val_ce, _ = stu(xv[:,:-1], xv[:,1:])
            stu.train()
            pbar.set_postfix(ce=f'{ce.item():.2f}', kl=f'{kl.item():.2f}',
                             val=f'{val_ce.item():.2f}', lr=f'{sched.get_last_lr()[0]:.1e}')
            torch.save({'model': stu.state_dict(), 'step': step}, f'{out}/distil_{step}.pt')

    torch.save({'model': stu.state_dict()}, f'{out}/distil_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--bsz', type=int, default=4)
    p.add_argument('--seq', type=int, default=1024)
    p.add_argument('--out', type=str, default='models/out-0.1b-distil')
    args = p.parse_args()
    train(cfg=None, total_steps=args.steps, lr=args.lr, bsz=args.bsz, seq=args.seq, out=args.out)
