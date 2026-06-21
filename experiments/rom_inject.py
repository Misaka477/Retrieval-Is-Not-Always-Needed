"""Latent ROM 注入实验: 检索 c_kv → 投影为 K/V → prepend 到 attention"""
import torch, math
from rina.model_a import RINA_A, RINA_A_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F, numpy as np

device='cuda'; torch.manual_seed(42)
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

# 1. 加载 Route A v3
cfg=RINA_A_Config(vocab_size=50257,block_size=512,use_int4=True)
m=RINA_A(cfg).to(device); m.eval()
sd=torch.load('models/out-rina-a-v3/rina-gen5-route-a-v3.pt',map_location=device,weights_only=False)['model']
m.load_state_dict(sd,strict=False)

# 2. 构建一个小的 ROM: 几段知识的 latent 索引
knowledge = [
    'Paris is the capital of France. It is located in Europe and is known for the Eiffel Tower.',
    'Machine learning is a subset of artificial intelligence that enables computers to learn from data.',
    'The CPU cache hierarchy consists of L1, L2, and L3 caches, each with different speeds and sizes.',
]

# Precompute latents for ROM
rom_latents = []  # list of [d_c] tensors
for text in knowledge:
    ids=tok.encode(text)[:64]
    x=torch.tensor([ids],device=device)
    with torch.no_grad():
        _,_,lats=m(x,x)
    # 取中间层 (layer 5) 的 latent, 平均
    lat=lats[0,5].mean(0).detach()  # [d_c]
    rom_latents.append(lat)

rom = F.normalize(torch.stack(rom_latents), dim=-1)  # [n_docs, d_c]
print(f'ROM: {len(rom)} docs, {rom.shape[1]}-dim latents')

# 3. 定义 ROM 增强的 forward
def forward_with_rom(model, idx, rom_latents, rom_kv, k=2):
    """在指定层注入 ROM K/V"""
    B,T=idx.shape
    h=model.transformer.drop(model.transformer.wte(idx))
    for layer_idx, block in enumerate(model.transformer.h):
        ln=block.ln1(h)
        d=block.attn
        cq=d.q_norm(d.w_dqkv(ln))
        qc=d.w_uq(cq).view(B,T,d.n_head,d.d_h).transpose(1,2)
        kc=d.w_uk(cq).view(B,T,d.n_kv,d.d_h).transpose(1,2)
        v=d.w_k2v(d.w_uk(cq)).view(B,T,d.n_kv,d.d_h).transpose(1,2)
        if d.use_qu:
            # 简化: 不量化（实验性质）
            pass
        qr=d.w_qr(ln).view(B,T,d.n_head,d.d_hr).transpose(1,2)
        kr=d.w_kr(ln).view(B,T,d.n_kv,d.d_hr).transpose(1,2)
        qr,kr=d.rope_q(qr),d.rope(kr)
        if d.n_rep>1:
            kc=kc.repeat_interleave(d.n_rep,1);v=v.repeat_interleave(d.n_rep,1);kr=kr.repeat_interleave(d.n_rep,1)
        
        # ROM 注入: 只在中间层 (4-7) 注入
        if 4<=layer_idx<=7 and rom_kv is not None:
            k_rom, v_rom = rom_kv  # [1, n_rom, H, d_h] each
            # prepend ROM K/V 到序列前面
            kc = torch.cat([k_rom.expand(B,-1,-1,-1), kc], dim=2)
            v = torch.cat([v_rom.expand(B,-1,-1,-1), v], dim=2)
            # ROM KV 没有 RoPE, 用 0 填充 kr
            kr_rom = torch.zeros(B, d.n_head, len(rom_latents), d.d_hr, device=x.device)
            kr = torch.cat([kr_rom, kr], dim=2)
        
        q=torch.cat([qc,qr],-1);k=torch.cat([kc,kr],-1)
        a=(q@k.transpose(-2,-1))/math.sqrt(q.size(-1))
        mask=torch.triu(torch.full((a.size(-2),a.size(-1)),float('-inf'),device=x.device),diagonal=1)
        a=a+mask.unsqueeze(0).unsqueeze(0)
        a=F.softmax(a,-1);y=a@v
        y=y.transpose(1,2).contiguous().view(B,T,-1)
        h=h+d.resid_drop(d.c_proj(y))+block.mlp(block.ln2(h))
    h=model.transformer.ln_f(h)
    return model.lm_head(h)

# 4. 预计算 ROM 的 K/V
def compute_rom_kv(model, rom_latents, layer_idx=5):
    """把 ROM latent 投影成 K/V"""
    d=model.transformer.h[layer_idx].attn
    cq=rom_latents.unsqueeze(0)  # [1, n_docs, d_c]
    with torch.no_grad():
        kc=d.w_uk(cq).view(1,-1,d.n_kv,d.d_h).transpose(1,2)  # [1, H, n_docs, d_h]
        v=d.w_k2v(d.w_uk(cq)).view(1,-1,d.n_kv,d.d_h).transpose(1,2)
        if d.n_rep>1:
            kc=kc.repeat_interleave(d.n_rep,1)
            v=v.repeat_interleave(d.n_rep,1)
    return kc, v  # [1, H, n_docs, d_h]

rom_kv=compute_rom_kv(m, rom)

# 5. 对比生成
prompt='The capital of France is'
ids=tok.encode(prompt)[:16]
x=torch.tensor([ids],device=device)

# Baseline (无 ROM)
x_base=x.clone()
with torch.no_grad():
    for _ in range(20):
        h=m.transformer.drop(m.transformer.wte(x_base))
        for b in m.transformer.h:
            ln=b.ln1(h);a,_=b.attn(ln);h=h+a+b.mlp(b.ln2(h))
        h=m.transformer.ln_f(h)
        l=m.lm_head(h[:,-1:])
        p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
        x_base=torch.cat([x_base,torch.multinomial(p,1)],1)
print(f'Baseline: {tok.decode(x_base[0].tolist())}')

# ROM 注入
x_rom=x.clone()
with torch.no_grad():
    for _ in range(20):
        l=forward_with_rom(m,x_rom,rom,rom_kv,k=2)
        p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
        x_rom=torch.cat([x_rom,torch.multinomial(p,1)],1)
print(f'With ROM: {tok.decode(x_rom[0].tolist())}')
