"""LoRA SFT: freeze base weights, train adapters. Prevents forgetting."""
import os, json, argparse, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rina.model_jamba_qw3 import RINA_Jamba_QW3, QW3_Config, Q4BlockLinear
from transformers import AutoTokenizer

device = 'cuda'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(
    os.path.join(BASE, 'models/teacher/LLM-Research/Llama-3___2-1B-Instruct'),
    local_files_only=True)
IGNORE = -1

# ── LoRA module ──

class LoRA(nn.Module):
    """Low-rank adapter attached to any PyTorch module with .weight."""
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r
        if hasattr(linear, 'in_dim'):
            in_dim, out_dim = linear.in_dim, linear.out_dim
        else:
            in_dim, out_dim = linear.in_features, linear.out_features
        self.A = nn.Parameter(torch.randn(in_dim, r) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, out_dim))
        linear.weight.requires_grad = False
        if linear.bias is not None: linear.bias.requires_grad = False
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.A) @ self.B * self.scale
        return base + lora

def add_lora(model, r=8, alpha=16):
    """Replace all nn.Linear in model with LoRA-wrapped versions."""
    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Q4BlockLinear)) and 'lm_head' not in name and 'norm_last' not in name and 'ln' not in name:
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            lora = LoRA(module, r, alpha).to(device)
            setattr(parent, parts[-1], lora)
            lora_layers.append(lora)
    
    # Get only LoRA parameters (base weights are frozen by LoRA.__init__)
    lora_params = [p for l in lora_layers for p in [l.A, l.B]]
    print(f'LoRA: {sum(p.numel() for p in lora_params)/1e3:.0f}K params')
    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable: {tot/1e3:.0f}K / {sum(p.numel() for p in model.parameters())/1e6:.0f}M total')
    return lora_params

# ── Data ──

def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d['text']
            if '\nAssistant: ' not in text: continue
            prompt, comp = text.split('\nAssistant: ', 1)
            prompt += '\nAssistant: '; comp = ' ' + comp
            p_ids = tok.encode(prompt); c_ids = tok.encode(comp)
            if 10 < len(p_ids) + len(c_ids) <= 512 and len(c_ids) > 10:
                rows.append((p_ids, c_ids))
    return rows

def collate(p_ids, c_ids, max_seq=512):
    labels = p_ids + c_ids
    input_ids = labels[:max_seq]
    labels = [IGNORE] * len(p_ids) + c_ids; labels = labels[:max_seq]
    if len(input_ids) < max_seq:
        input_ids += [0] * (max_seq - len(input_ids))
        labels += [IGNORE] * (max_seq - len(labels))
    return input_ids, labels

# ── Train ──

def train(steps=20000, lr=1e-4):
    magic = load_data(os.path.join(BASE, 'data/magic_oss_en.jsonl'))
    print(f'Data: {len(magic)} Magic')
    
    c = QW3_Config(vocab_size=128256,block_size=512,use_int4=False,n_embd=640,
        n_layer=16,n_head=10,n_kv_heads=5,d_c=160,head_dim=64,
        sparse_k=16,sparse_window=32,sparse_local_w=4,
        ssm_steps=3,quant_mode='q4k_q2v',ssm_qbits=0,weight_bits=16,embed_quant=False)
    model = RINA_Jamba_QW3(c).to(device)
    sd = torch.load(os.path.join(BASE, 'models/out-rina-jamba-qw3-p1/jambaqw3_140500.pt'),
                    map_location='cpu', weights_only=True)
    model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
    
    lora_params = add_lora(model)
    
    out = os.path.join(BASE, 'models/out-jamba-lora')
    os.makedirs(out, exist_ok=True)
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    log_file = open(os.path.join(out, 'train.log'), 'w')
    pbar = tqdm(range(steps))
    gn_sum=gn_cnt=0
    
    for step in pbar:
        model.train()
        p_ids, c_ids = magic[np.random.randint(len(magic))]
        x_ids, labels = collate(p_ids, c_ids)
        x = torch.tensor(x_ids).long().to(device).unsqueeze(0)
        y = torch.tensor(labels).long().to(device).unsqueeze(0)
        if not torch.any(y != IGNORE): continue
        _, ce = model(x, y)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step(); opt.zero_grad()
        gn_sum+=gn.item(); gn_cnt+=1
        
        if step%200==0:
            model.eval()
            with torch.no_grad():
                p2,c2 = magic[np.random.randint(len(magic))]
                vx = torch.tensor(collate(p2,c2)[0]).unsqueeze(0).to(device)
                vy = torch.tensor(collate(p2,c2)[1]).unsqueeze(0).to(device)
                _, vce = model(vx, vy)
            msg = f'step={step} ce={ce.item():.4f} val={vce.item():.4f} gn={gn_sum/max(gn_cnt,1):.2f}'
            log_file.write(msg+'\n'); log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.4f}', val=f'{vce.item():.4f}')
    
    # Save model state (base + LoRA) — LoRA params have module names like 'transformer.h.0.path.w_dq.A'
    torch.save(model.state_dict(), f'{out}/lora_final.pt')
    log_file.close()
    print('Done')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20000)
    args = p.parse_args()
    train(steps=args.steps)
