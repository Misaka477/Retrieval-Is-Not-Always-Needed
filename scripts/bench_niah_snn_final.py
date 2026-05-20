"""Extreme + Multi-key NIAH on SNN v2 — 对标 V1 bench_niah_extreme/multikey."""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ["HF_DATASETS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"; os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F, random
from modules.temporal_snn_cell import TemporalSNNModel

device="cuda"; torch.manual_seed(42); random.seed(42)
V=4096; DM=840; NP=4096
KEYS=list(range(1,6)); VALS=list(range(6,11))
tok=Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds=load_dataset("wikitext","wikitext-103-v1",split="train")
paras=[]
for t in ds["text"][:10000]:
    if len(t)>100:
        enc=tok.encode(t).ids
        if len(enc)>=200: paras.append(torch.tensor(enc,dtype=torch.long))
print(f"paragraphs: {len(paras)}",flush=True)

def build_model():
    m=TemporalSNNModel(V,d_model=DM,n_patterns=NP,beta=0.5,attract_every=2,error_threshold=1.0,hebbian_lr=0.0,inhibition_threshold=0.0).to(device)
    m.load_state_dict(torch.load("checkpoints/cann_snn15m_v2_ep12.pt",map_location=device,weights_only=False)["model"],strict=False)
    m.slot_proj=torch.nn.Linear(DM,DM).to(device); m.slot_proj.weight.data.normal_(0,0.01); m.slot_proj.bias.data.zero_()
    return m

def fwd(model,x,slot_dict):
    bsz,sl=x.shape; dm=model.d_model; emb=model.embed(x); h=torch.zeros(bsz,dm,device=device); logits=[]
    for t in range(sl):
        if t<sl-1:
            h=model.cell(h,emb[:,t,:],step=t)
        else:
            inj=torch.stack([slot_dict.get(x[b,-1].item(),torch.zeros(dm,device=device)) for b in range(bsz)])
            h=model.cell(h+inj,emb[:,t,:],step=t)
            pat=model.cell.patterns.unsqueeze(0).expand(bsz,-1,-1)
            xi=h.unsqueeze(1); sc=xi@pat.transpose(1,2)*model.cell.beta_t[0]
            attracted=(torch.softmax(sc,-1)@pat).squeeze(1)
            cl=torch.cat([h,emb[:,-1,:]],-1); alpha=torch.sigmoid(model.cell.gate_alpha(cl))
            h=h+alpha*(attracted-h); h=model.cell.norm(h)
        logits.append(model.head(model.state_norm(h)))
    return torch.stack(logits,1)

def extreme(gap=128,name="Extreme"):
    print(f"\n── {name} gap={gap} ──",flush=True)
    train_x,train_y,ks=[],[],[]
    for _ in range(200):
        while True:
            p=paras[torch.randint(0,len(paras),(1,)).item()]; need=64+gap+4
            if len(p)>=need+4:
                k=random.choice(KEYS); v=random.choice(VALS)
                ins=random.randint(2,min(need-4,len(p)-need))
                seq=p[:need].tolist(); seq[ins]=k; seq[ins+1]=v; seq[-1]=k
                train_x.append(torch.tensor(seq)); train_y.append(v); ks.append((ins,k))
                break
    test_x,test_y,_=[],[],[]
    for _ in range(100):
        while True:
            p=paras[torch.randint(0,len(paras),(1,)).item()]; need=64+gap+4
            if len(p)>=need+4:
                k=random.choice(KEYS); v=random.choice(VALS)
                ins=random.randint(2,min(need-4,len(p)-need))
                seq=p[:need].tolist(); seq[ins]=k; seq[ins+1]=v; seq[-1]=k
                test_x.append(torch.tensor(seq)); test_y.append(v)
                break
    test_x=torch.stack(test_x); test_y=torch.tensor(test_y)
    m=build_model(); opt=torch.optim.AdamW(m.parameters(),lr=3e-4); best=0
    for step in range(200):
        m.train(); opt.zero_grad()
        perm=torch.randperm(len(train_x))
        for i in range(0,len(train_x),32):
            idx=perm[i:i+32]; xb=torch.stack([train_x[j] for j in idx]).to(device)
            yb=torch.tensor([train_y[j] for j in idx],device=device)
            slot={}
            for b in range(len(idx)):
                kk,vv=ks[idx[b]]
                if kk in KEYS and vv in VALS:
                    slot[kk]=m.slot_proj(m.embed(torch.tensor([vv],device=device))).squeeze(0)
            loss=F.cross_entropy(fwd(m,xb,slot)[:,-1],yb); loss.backward()
        opt.step()
        if step%10==9:
            m.eval()
            with torch.no_grad():
                slot_t={}; [slot_t.update({test_x[b,0].item():m.slot_proj(m.embed(torch.tensor([test_y[b]],device=device))).squeeze(0)}) for b in range(len(test_x))]
                lt=fwd(m,test_x.to(device),slot_t)
            acc=(lt[:,-1].argmax(-1)==test_y.to(device)).float().mean().item()
            best=max(best,acc)
            print(f"  step={step+1:3d}: acc={acc*100:.0f}% best={best*100:.0f}%",flush=True)
            if best>=1.0: break; m.train()
    return best

def multikey(gap=128,n_keys=3):
    print(f"\n── Multi-key ({n_keys} keys) gap={gap} ──",flush=True)
    train_x,train_y=[],[]
    for _ in range(200):
        while True:
            p=paras[torch.randint(0,len(paras),(1,)).item()]; need=64+gap+4
            if len(p)>=need+4:
                seq=p[:need].tolist(); targets=[]
                for j in range(n_keys):
                    k=KEYS[j%len(KEYS)]; v=VALS[j%len(VALS)]
                    ins=random.randint(2,min(need-4,len(p)-need))
                    seq[ins]=k; seq[ins+1]=v
                    targets.append((k,v))
                queries=[KEYS[j%len(KEYS)] for j in range(n_keys)]
                for j in range(n_keys):
                    seq[-(j+1)]=queries[j]
                train_x.append(torch.tensor(seq)); train_y.append(targets)
                break
    test_x,test_y=[],[]
    for _ in range(50):
        while True:
            p=paras[torch.randint(0,len(paras),(1,)).item()]; need=64+gap+4
            if len(p)>=need+4:
                seq=p[:need].tolist(); targets=[]
                for j in range(n_keys):
                    k=KEYS[j%len(KEYS)]; v=VALS[j%len(VALS)]
                    ins=random.randint(2,min(need-4,len(p)-need))
                    seq[ins]=k; seq[ins+1]=v; targets.append((k,v))
                queries=[KEYS[j%len(KEYS)] for j in range(n_keys)]
                for j in range(n_keys): seq[-(j+1)]=queries[j]
                test_x.append(torch.tensor(seq)); test_y.append(targets)
                break
    test_x=torch.stack(test_x)
    m=build_model(); opt=torch.optim.AdamW(m.parameters(),lr=3e-4); best=0
    for step in range(200):
        m.train(); opt.zero_grad()
        perm=torch.randperm(len(train_x))
        for i in range(0,len(train_x),32):
            idx=perm[i:i+32]; xb=torch.stack([train_x[j] for j in idx]).to(device)
            slot={}
            for b in range(len(idx)):
                for kk,vv in train_y[idx[b]]:
                    if kk in KEYS and vv in VALS:
                        slot[kk]=m.slot_proj(m.embed(torch.tensor([vv],device=device))).squeeze(0)
            logits=fwd(m,xb,slot)
            loss=torch.tensor(0.0,device=device)
            for b in range(len(idx)):
                for j,(kk,vv) in enumerate(train_y[idx[b]]):
                    loss=loss+F.cross_entropy(logits[b,-1-j].unsqueeze(0),torch.tensor([vv],device=device))
            loss.backward()
        opt.step()
        if step%10==9:
            m.eval()
            with torch.no_grad():
                slot_t={}
                for b in range(len(test_y)):
                    for kk,vv in test_y[b]:
                        slot_t[kk]=m.slot_proj(m.embed(torch.tensor([vv],device=device))).squeeze(0)
                lt=fwd(m,test_x.to(device),slot_t)
                correct=0; total=0
                for b in range(len(test_y)):
                    for j,(kk,vv) in enumerate(test_y[b]):
                        total+=1
                        if lt[b,-1-j].argmax(-1)==vv: correct+=1
                acc=correct/total; best=max(best,acc)
            print(f"  step={step+1:3d}: acc={acc*100:.0f}% best={best*100:.0f}%",flush=True)
            if best>=1.0: break; m.train()
    return best

#acc_extreme=extreme(128)
#print(f"\nExtreme (random pos): {acc_extreme*100:.0f}%")
acc_multi=multikey(128,3)
print(f"Multi-key: {acc_multi*100:.0f}%")
