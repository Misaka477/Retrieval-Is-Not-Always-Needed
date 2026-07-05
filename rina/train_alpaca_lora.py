"""Alpaca LoRA SFT: single-turn instruction format, no User/Assistant tags, freeze base."""
import os, torch, torch.nn as nn, numpy as np
from tqdm import tqdm
from rina.model_jamba_qw3 import RINA_Jamba_QW3, QW3_Config, Q4BlockLinear
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(
    os.path.join(BASE, 'models/teacher/LLM-Research/Llama-3___2-1B-Instruct'),
    local_files_only=True)

class LoRA(nn.Module):
    def __init__(self, linear, r=8, a=16):
        super().__init__()
        self.linear = linear; self.scale = a / r
        if hasattr(linear, 'in_dim'): i, o = linear.in_dim, linear.out_dim
        else: i, o = linear.in_features, linear.out_features
        self.A = nn.Parameter(torch.randn(i, r) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, o))
        linear.weight.requires_grad = False
    def forward(self, x):
        return self.linear(x) + (x @ self.A) @ self.B * self.scale

def add_lora(model, r=8):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, (nn.Linear, Q4BlockLinear)) and 'lm_head' not in name:
            parent = model
            for p in name.split('.')[:-1]:
                parent = getattr(parent, p)
            setattr(parent, name.split('.')[-1], LoRA(mod, r).cuda())

def main():
    from datasets import load_dataset
    ds = load_dataset('tatsu-lab/alpaca', split='train')
    rows = []
    for ex in ds:
        text = f"Instruction: {ex['instruction']}\nResponse: {ex['output']}"
        ids = tok.encode(text)
        if 20 < len(ids) <= 512:
            rows.append(ids)
    print(f'Alpaca: {len(rows)}')

    c = QW3_Config(vocab_size=128256, block_size=512, use_int4=False,
        n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160, head_dim=64,
        sparse_k=16, sparse_window=32, sparse_local_w=4,
        ssm_steps=3, quant_mode='q4k_q2v', ssm_qbits=0, weight_bits=16, embed_quant=False)
    m = RINA_Jamba_QW3(c).cuda()
    sd = torch.load(os.path.join(BASE, 'models/out-rina-jamba-qw3-p1/jambaqw3_140500.pt'),
                    map_location='cpu', weights_only=True)
    m.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)

    add_lora(m)
    lora_params = [p for p in m.parameters() if p.requires_grad]
    print(f'LoRA: {sum(p.numel() for p in lora_params) / 1e3:.0f}K params')

    out_dir = os.path.join(BASE, 'models/out-jamba-alpaca')
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(lora_params, lr=3e-5)
    log_file = open(os.path.join(out_dir, 'train.log'), 'w')
    pbar = tqdm(range(5000))

    for step in pbar:
        m.train()
        ids = torch.tensor(rows[np.random.randint(len(rows))]).long().cuda()
        x = ids[:-1].unsqueeze(0)
        y = ids[1:].unsqueeze(0)
        _, ce = m(x, y)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        opt.zero_grad()
        if step % 500 == 0:
            msg = f'step={step} ce={ce.item():.4f} gn={gn:.2f}'
            log_file.write(msg + '\n')
            log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.4f}')

    torch.save(m.state_dict(), os.path.join(out_dir, 'lora_alpaca.pt'))
    log_file.close()
    print('Done')

if __name__ == '__main__':
    main()
