"""RINA Gen 6 — Jamba 生成测试。用法：
  python3 -m rina.gen                           # Jamba random init
  python3 -m rina.gen --jamba path.pt           # 加载 Jamba checkpoint
  python3 -m rina.gen --prompt "Hello" --temp 0.3 --topk 20 --rep 5.0
  python3 -m rina.gen --jamba path.pt --lq       # LSC q4 SSM 版
  python3 -m rina.gen --jamba path.pt --qw       # QW 极压版
"""
import torch, torch.nn.functional as F, argparse, os, sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_jamba(ckpt_path, model_type='base'):
    from rina.model_jamba import RINA_Jamba, RJ_Config
    from rina.model_jamba_lq import RINA_Jamba_LQ, RJLQ_Config
    from rina.model_jamba_qw import RINA_Jamba_QW, QW_Config

    base_cfg = dict(vocab_size=128256, block_size=512, use_int4=True,
        n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160, head_dim=64,
        sparse_k=16, sparse_window=32, sparse_local_w=4, ssm_steps=3)

    if model_type == 'lq':
        cfg = RJLQ_Config(**base_cfg, quant_mode='q4k_q2v', ssm_qbits=0)
        model = RINA_Jamba_LQ(cfg).to(device).eval()
    elif model_type == 'qw':
        cfg = QW_Config(**base_cfg, quant_mode='q2k_q1v', ssm_qbits=4, weight_bits=4)
        model = RINA_Jamba_QW(cfg).to(device).eval()
    else:
        cfg = RJ_Config(**base_cfg)
        model = RINA_Jamba(cfg).to(device).eval()

    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
        print(f'Loaded: {ckpt_path}')
    else:
        print('Random init')
    return model

def generate(model, prompts, steps=64, temp=0.3, top_k=20, rep=5.0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        'models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
    tok.pad_token = tok.eos_token

    for pr in prompts:
        ids = tok.encode(pr, return_tensors='pt').to(device)
        x = ids.clone()
        generated_ids = set()
        for _ in range(steps):
            with torch.no_grad():
                l, _ = model(x)
            logits = l[0, -1].float() / temp
            if top_k > 0:
                vv, _ = torch.topk(logits, top_k)
                logits[logits < vv[-1]] = -float('Inf')
            p = F.softmax(logits, -1)
            if rep > 1.0 and generated_ids:
                for gid in list(generated_ids):
                    if gid < p.size(-1):
                        p[gid] /= rep
                p = p / p.sum()
            nxt = torch.multinomial(p.unsqueeze(0), 1)
            generated_ids.add(nxt.item())
            x = torch.cat([x, nxt], 1)
        print(f'\nP: {pr}')
        print(f'G: {tok.decode(x[0].tolist())}\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--jamba', type=str, default=None, help='Jamba checkpoint')
    p.add_argument('--lq', action='store_true')
    p.add_argument('--qw', action='store_true')
    p.add_argument('--prompt', type=str, default=None)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--temp', type=float, default=0.3)
    p.add_argument('--topk', type=int, default=20)
    p.add_argument('--rep', type=float, default=5.0)
    args = p.parse_args()

    model_type = 'qw' if args.qw else 'lq' if args.lq else 'base'
    model = load_jamba(args.jamba, model_type)
    prompts = [args.prompt] if args.prompt else [
        'The capital of France is', 'The meaning of life is',
        'Once upon a time,', 'In the theory of relativity,']
    generate(model, prompts, args.steps, args.temp, args.topk, args.rep)
