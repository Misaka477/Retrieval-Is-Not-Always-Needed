#!/usr/bin/env bash
# Latent ROM 演示: 写入知识 → 低置信度时自动检索注入
cd "$(dirname "$0")"

python3 -c "
import torch
from rina.rom import LatentROM
from rina.model_cf import RINA_CF, RINA_CF_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'; torch.manual_seed(42)
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg=RINA_CF_Config(vocab_size=50257,block_size=512,use_int4=True)
m=RINA_CF(cfg).to(device); m.eval()
m.load_state_dict(torch.load('models/out-rina-cf/rina_cf_final.pt',
    map_location='cpu',weights_only=False)['model'],strict=False)

# 创建 ROM 并写入一段模型没训过的虚构知识
rom=LatentROM(dim=128)
n=rom.write_text(m,tok,'In the land of Voria, the sacred flower is called the Lumina Bloom. '
    'It glows blue at night and is used in healing rituals.',layer_idx=5)
print(f'ROM: {n} tokens, {len(rom)} entries')
print(f'Token Lumina id: {tok.encode(\" Lumina\")[0]}')

# 生成对比
prompt='In Voria, the sacred flower is called the'
ids=tok.encode(prompt)[:12]
x=torch.tensor(ids,device=device).unsqueeze(0)

def generate(use_rom=False):
    xi=x.clone()
    with torch.no_grad():
        for _ in range(20):
            h=m.transformer.drop(m.transformer.wte(xi))
            for li,blk in enumerate(m.transformer.h):
                ln=blk.ln1(h)
                kv=None
                if use_rom and 4<=li<=7 and len(rom)>0:
                    cq=blk.l2.q_norm(blk.l2.w_dqkv(ln))
                    conf=blk.conf_head(cq).mean().item()
                    if conf>cfg.confidence_thresholds[2]:
                        kv=rom.get_kv(m,li,cq[0,-1],k=4)
                        kv=kv if kv[0] is not None else None
                out,_=blk.l2(ln,kv)
                h=h+out+blk.mlp(blk.ln2(h))
            hf=m.transformer.ln_f(h)
            l=m.lm_head(hf[:,-1:])
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            xi=torch.cat([xi,torch.multinomial(p,1)],1)
    return tok.decode(xi[0].tolist())

print(f'Baseline: {generate(False)}')
print(f'With ROM: {generate(True)}')
" 2>&1 | grep -v 'RINA_CF:'
