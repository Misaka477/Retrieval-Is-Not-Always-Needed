#!/usr/bin/env bash
# 置信度调度详细演示: 逐层路由 + 逐步骤生成
cd "$(dirname "$0")"

python3 -c "
import torch
from rina.model_cf import RINA_CF, RINA_CF_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg=RINA_CF_Config(vocab_size=50257,block_size=512,use_int4=True)
m=RINA_CF(cfg).to(device); m.eval()
m.load_state_dict(torch.load('models/out-rina-cf/rina_cf_final.pt',
    map_location='cpu',weights_only=False)['model'],strict=False)

th=cfg.confidence_thresholds
TAG=['L1','L3','L2','ROM']
SEED=42

prompts = [
    ('代码', '#!/usr/bin/env python3'),
    ('自然语言', 'The capital of France is'),
]

for label, prompt in prompts:
    torch.manual_seed(SEED)
    ids=tok.encode(prompt)[:12]
    x=torch.tensor(ids,device=device).unsqueeze(0)

    print(f'═══ [{label}] {prompt} ═══')
    total={'L1':0,'L3':0,'L2':0,'ROM':0}

    with torch.no_grad():
        for step in range(20):
            h=m.transformer.drop(m.transformer.wte(x))
            step_routes=[]
            for li,blk in enumerate(m.transformer.h):
                ln=blk.ln1(h)
                cq=blk.l2.q_norm(blk.l2.w_dqkv(ln))
                c=blk.conf_head(cq).mean().item()
                if c<th[0]: r='L1'
                elif c<th[1]: r='L3'
                elif c<th[2]: r='L2'
                else: r='ROM'
                step_routes.append(r)
                if r=='L1': out=blk.l1(ln)
                elif r=='L3': out,_=blk.l3(ln)
                else: out,_=blk.l2(ln)
                h=h+out+blk.mlp(blk.ln2(h))

            # 解码当前最后一个 token
            curr=tok.decode(x[0].tolist())
            hf=m.transformer.ln_f(h)
            l=m.lm_head(hf[:,-1:])
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            nxt=torch.multinomial(p,1)
            next_tok=tok.decode(nxt[0].tolist())

            # 累计
            for r in step_routes: total[r]+=1

            # 显示前5步的详细路由，之后只显示累计
            if step<5:
                route_str=' '.join(step_routes)
                print(f'  Step {step:2d} [{next_tok:>8s}] {route_str}')
            elif step==5:
                print(f'  ... (后续步骤不再逐层显示)')

    tot=sum(total.values())
    print(f'  累计: L1={total[\"L1\"]}({total[\"L1\"]/tot*100:.0f}%) L3={total[\"L3\"]}({total[\"L3\"]/tot*100:.0f}%) L2={total[\"L2\"]}({total[\"L2\"]/tot*100:.0f}%) ROM={total[\"ROM\"]}({total[\"ROM\"]/tot*100:.0f}%)')
    print(f'  生成: {tok.decode(x[0].tolist())}')
    print()
" 2>&1 | grep -v 'RINA_CF:'
