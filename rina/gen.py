"""RINA 生成测试。用法：
  python3 -m rina.gen                    # 随机初始化（看 loss）
  python3 -m rina.gen --load path.pt     # 加载权重看生成
"""
import torch, torch.nn.functional as F, argparse, os
from .model import RINA, RINAConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate(cfg, ckpt_path=None, prompts=None, steps=40, temp=0.8):
    model = RINA(cfg).to(device); model.eval()
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
        print(f'Loaded: {ckpt_path}')
    else:
        print(f'Random init (loss≈11.0)')

    if prompts is None:
        prompts = ["The capital of France is", "The meaning of life is"]

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('checkpoints/gpt2_tokenizer')
    tok.pad_token = tok.eos_token

    for pr in prompts:
        ids = tok.encode(pr)[:16]
        x = torch.tensor([ids], device=device)
        for _ in range(steps):
            l, _ = model(x)
            p = F.softmax(l[:, -1].float() / temp, -1)
            p[0, 0] = 0
            x = torch.cat([x, torch.multinomial(p, 1)], 1)
        print(f'P: {pr}')
        print(f'G: {tok.decode(x[0].tolist())}\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--load', type=str, default=None)
    p.add_argument('--seq', type=int, default=512)
    p.add_argument('--158', action='store_true')
    p.add_argument('--int4', action='store_true')
    args = p.parse_args()
    cfg = RINAConfig(block_size=args.seq, use_158=args.__dict__['158'], use_int4=args.int4)
    generate(cfg, ckpt_path=args.load)
