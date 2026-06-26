"""测试 L3X 滑窗稀疏 vs 全量 attention"""
import torch
from rina.model_l3x import RINA_L3X, RINA_L3X_Config, SparseIndexManager
from transformers import AutoTokenizer
import torch.nn.functional as F

device='cuda'; torch.manual_seed(42)
tok=AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token=tok.eos_token

cfg=RINA_L3X_Config(vocab_size=128256,block_size=512,use_int4=True,
    n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160,sparse_k=8,sparse_window=32)
m=RINA_L3X(cfg).to(device); m.eval()
sd=torch.load('models/out-0.1b-ce/ce_298000.pt',map_location='cpu',weights_only=False)['model']
own=m.state_dict()
for k,v in sd.items():
    if k in own and own[k].shape==v.shape:
        own[k].data.copy_(v.to(device))

prompt='The meaning of life is'
ids=tok.encode(prompt)[:12]

for label,window in [('Full',0),('W=8',8),('W=16',16),('W=32',32)]:
    mgr=SparseIndexManager(window=window,k=8,local_w=4) if window>0 else None
    x=torch.tensor(ids,device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(20):
            h=m.transformer.drop(m.transformer.wte(x))
            for li,b in enumerate(m.transformer.h):
                ln=b.ln1(h)
                if mgr and b.attn.sparse and li==12:
                    cq=b.attn.q_norm(b.attn.w_dqkv(ln))
                    mgr.step_forward(cq,x.size(1),x.device)
                a,_=b.attn(ln,mgr if b.attn.sparse else None)
                h=h+a+b.mlp(b.ln2(h))
            hf=m.transformer.ln_f(h);l=m.lm_head(hf[:,-1:])
            p=F.softmax(l[0,-1]/0.5,-1);p[0]=0
            vv,_=torch.topk(p,50);p[p<vv[-1]]=0
            x=torch.cat([x,torch.multinomial(p,1).unsqueeze(0)],1)
    print(f'  {label}: {tok.decode(x[0].tolist())[:100]}')
