"""Latent ROM 验证: 注入模型未训练过的新知识，看能否影响生成"""
import sys; sys.path.insert(0,'.')
import torch, math, numpy as np
from rina.model_a import RINA_A, RINA_A_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'; torch.manual_seed(42)
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

# 用 best baseline
cfg=RINA_A_Config(vocab_size=50257,block_size=512,use_int4=True)
m=RINA_A(cfg).to(device); m.eval()
sd=torch.load('models/out-quant/rina-gen5-baseline-int4.pt',map_location=device,weights_only=False)['model']
m.load_state_dict(sd,strict=False)

# ROM 知识: 一个模型绝对没训过的虚构设定
rom_fact = "In the land of Voria, the sacred flower is called the Lumina Bloom. It glows blue at night and is used in healing rituals."
query = "In Voria, the sacred flower is called the"

# 预计算 ROM 的 per-token latent
ids=tok.encode(rom_fact)[:48]
x=torch.tensor([ids],device=device)
with torch.no_grad():
    _,_,lats=m(x,x)
rom_lats=lats[0,5].detach()  # [T, d_c], layer 5
print(f'ROM: {len(ids)} tokens indexed')
print(f'Fact: {tok.decode(ids)}')
print()

# 检查模型本来知不知道" Lumina"
lumina_id=tok.encode(' Lumina')[0]
print(f'Token " Lumina" id={lumina_id}, exists in vocab? {lumina_id < cfg.vocab_size}')

# 预计算 ROM K/V
d=m.transformer.h[5].attn
cq=rom_lats.unsqueeze(0)  # [1, T, d_c]
with torch.no_grad():
    k_rom=d.w_uk(cq).view(1,-1,d.n_kv,d.d_h).transpose(1,2)  # [1, H, T, d_h]
    v_rom=d.w_k2v(d.w_uk(cq)).view(1,-1,d.n_kv,d.d_h).transpose(1,2)
    if d.n_rep>1:
        k_rom=k_rom.repeat_interleave(d.n_rep,1)
        v_rom=v_rom.repeat_interleave(d.n_rep,1)

# 生成函数
def generate(prompt, use_rom=False, steps=20):
    ids=tok.encode(prompt)[:16]
    x=torch.tensor(ids,device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps):
            B,T=x.shape
            h=m.transformer.drop(m.transformer.wte(x))
            for li,block in enumerate(m.transformer.h):
                ln=block.ln1(h); d=block.attn
                cq=d.q_norm(d.w_dqkv(ln))
                qc=d.w_uq(cq).view(B,T,d.n_head,d.d_h).transpose(1,2)
                kc=d.w_uk(cq).view(B,T,d.n_kv,d.d_h).transpose(1,2)
                v=d.w_k2v(d.w_uk(cq)).view(B,T,d.n_kv,d.d_h).transpose(1,2)
                qr=d.w_qr(ln).view(B,T,d.n_head,d.d_hr).transpose(1,2)
                kr=d.w_kr(ln).view(B,T,d.n_kv,d.d_hr).transpose(1,2)
                qr,kr=d.rope_q(qr),d.rope(kr)
                if d.n_rep>1:
                    kc=kc.repeat_interleave(d.n_rep,1);v=v.repeat_interleave(d.n_rep,1);kr=kr.repeat_interleave(d.n_rep,1)
                
                if use_rom and 4<=li<=7:
                    kr_rom=torch.zeros(1,d.n_head,rom_lats.shape[0],d.d_hr,device=x.device)
                    kc=torch.cat([k_rom.expand(B,-1,-1,-1),kc],dim=2)
                    v=torch.cat([v_rom.expand(B,-1,-1,-1),v],dim=2)
                    kr=torch.cat([kr_rom,kr],dim=2)
                
                q=torch.cat([qc,qr],-1);k=torch.cat([kc,kr],-1)
                a=(q@k.transpose(-2,-1))/math.sqrt(q.size(-1))
                mask=torch.triu(torch.full((a.size(-2),a.size(-1)),float('-inf'),device=x.device),diagonal=1)
                a=a+mask.unsqueeze(0).unsqueeze(0);a=F.softmax(a,-1);y=a@v
                y=y.transpose(1,2).contiguous().view(B,T,-1)
                h=h+d.resid_drop(d.c_proj(y))+block.mlp(block.ln2(h))
            h=m.transformer.ln_f(h)
            l=m.lm_head(h[:,-1:])
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            x=torch.cat([x,torch.multinomial(p,1)],1)
    return tok.decode(x[0].tolist())

print('=== Generation ===')
print(f'No ROM: {generate(query, use_rom=False)}')
print(f'With ROM: {generate(query, use_rom=True)}')

# 概率对比: " Lumina" 和 " Bloom" 
print()
print('=== Token probability comparison ===')
for target in [' Lumina', ' Bloom', ' healing', ' blue', ' Voria']:
    target_id=tok.encode(target)[0]
    ids=tok.encode(query)[:16]
    x=torch.tensor(ids,device=device).unsqueeze(0)
    T=x.size(1)
    
    def forward_one(prompt_ids, inject_rom=False):
        """单次前向, 只要最后一个位置的 logits"""
        B,seq_len=prompt_ids.shape
        h=m.transformer.drop(m.transformer.wte(prompt_ids))
        for li,block in enumerate(m.transformer.h):
            ln=block.ln1(h);d=block.attn
            cq=d.q_norm(d.w_dqkv(ln))
            qc=d.w_uq(cq).view(B,seq_len,d.n_head,d.d_h).transpose(1,2)
            kc=d.w_uk(cq).view(B,seq_len,d.n_kv,d.d_h).transpose(1,2)
            v=d.w_k2v(d.w_uk(cq)).view(B,seq_len,d.n_kv,d.d_h).transpose(1,2)
            qr=d.w_qr(ln).view(B,seq_len,d.n_head,d.d_hr).transpose(1,2)
            kr=d.w_kr(ln).view(B,seq_len,d.n_kv,d.d_hr).transpose(1,2)
            qr,kr=d.rope_q(qr),d.rope(kr)
            if d.n_rep>1:
                kc=kc.repeat_interleave(d.n_rep,1);v=v.repeat_interleave(d.n_rep,1);kr=kr.repeat_interleave(d.n_rep,1)
            if inject_rom and 4<=li<=7:
                kr_rom=torch.zeros(1,d.n_head,rom_lats.shape[0],d.d_hr,device=x.device)
                kc=torch.cat([k_rom,kc],dim=2);v=torch.cat([v_rom,v],dim=2);kr=torch.cat([kr_rom,kr],dim=2)
            nk=kc.size(2)
            q=torch.cat([qc,qr],-1);k=torch.cat([kc,kr],-1)
            a=(q@k.transpose(-2,-1))/math.sqrt(q.size(-1))
            mask=torch.triu(torch.full((seq_len,nk),float('-inf'),device=x.device),diagonal=1)
            a=a+mask.unsqueeze(0).unsqueeze(0);a=F.softmax(a,-1);y=a@v
            y=y.transpose(1,2).contiguous().view(B,seq_len,-1)
            h=h+d.resid_drop(d.c_proj(y[:,:seq_len]))+block.mlp(block.ln2(h))
        h=m.transformer.ln_f(h)
        return m.lm_head(h[:,-1:])
    
    with torch.no_grad():
        l_base=forward_one(x,inject_rom=False)
        p_base=F.softmax(l_base[0,-1],-1)[target_id].item()
        l_rom=forward_one(x,inject_rom=True)
        p_rom=F.softmax(l_rom[0,-1],-1)[target_id].item()
    
    delta=(p_rom-p_base)*100
    print(f'{target:>10s}: base={p_base*100:.2f}%  rom={p_rom*100:.2f}%  delta={delta:+.2f}%')
