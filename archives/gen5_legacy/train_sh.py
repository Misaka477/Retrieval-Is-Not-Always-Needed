"""Train SH: 硬路由训练 — 调度头选一路, 只训那一路"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_sh import RINA_SH, RINA_SH_Config

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_into(model, ckpt_path, src_sub, dst_sub):
    sd=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    state=sd['model'] if 'model' in sd else sd
    own=dict(model.named_parameters())
    n=0
    for k,v in state.items():
        target=k.replace(src_sub,dst_sub)
        if target in own and own[target].shape==v.shape:
            own[target].data.copy_(v);n+=1
    return n

def train(cfg, steps=3000, lr=1e-3, bsz=4, out='models/out-rina-sh'):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/english_pretrain.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10

    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_SH(cfg).to(device)
    own=dict(model.named_parameters())

    # 三路分别加载
    n1=load_into(model,'models/out-rina-c/rina-gen5-route-c.pt','inertia.','l1.')
    n2=load_into(model,'models/out-quant/rina-gen5-baseline-int4.pt','attn.','l2.')
    n3=load_into(model,'models/out-rina-a-v3/rina-gen5-route-a-v3.pt','attn.','l3.')
    print(f'Loaded: L1={n1} L2={n2} L3={n3} keys')

    # 解冻全部
    for p in model.parameters(): p.requires_grad=True
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable: {trainable} params (all)')

    backbone_params=[p for n,p in model.named_parameters() if 'scheduler' not in n]
    sched_params=[p for n,p in model.named_parameters() if 'scheduler' in n]
    opt=torch.optim.AdamW([
        {'params':backbone_params,'lr':lr*0.3,'weight_decay':0.01},
        {'params':sched_params,'lr':lr,'weight_decay':0.0},
    ],betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    pbar=tqdm(range(steps));t0=time.time()

    for cur_step in pbar:
        x=get_batch(cfg.block_size+1)
        l,ce,routes,choices=model(x[:,:-1],x[:,1:],return_all=True)
        loss=ce
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()

        if cur_step%500==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(cfg.block_size+1)
                lv,val_ce,_=model(xv[:,:-1],xv[:,1:])
                h=model.transformer.drop(model.transformer.wte(xv[:,:-1]))
                rts=[]
                with torch.no_grad():
                    for blk in model.transformer.h:
                        ln=blk.ln1(h);_,cq=blk.l3(ln)
                        rt=F.softmax(blk.scheduler(cq.detach()),dim=-1)
                        rts.append(rt.mean((0,1)))
                        wt=rt.mean(dim=(0,1));chosen=wt.argmax().item()
                        if chosen==0: o=blk.l1(ln)
                        elif chosen==1: o,_=blk.l2(ln)
                        else: o,_=blk.l3(ln)
                        h=h+o+blk.mlp(blk.ln2(h))
                route_dist=torch.stack(rts).mean(0).tolist()
            model.train()
            pbar.set_postfix(ce=f'{ce.item():.2f}',val=f'{val_ce:.2f}',
                             l1=f'{route_dist[0]:.2f}',l2=f'{route_dist[1]:.2f}',l3=f'{route_dist[2]:.2f}')
            torch.save({'model':model.state_dict(),'step':cur_step,'routes':route_dist},
                       f'{out}/rina_sh_{cur_step}.pt')

    torch.save({'model':model.state_dict()},f'{out}/rina_sh_final.pt')
    # 最终路由
    model.eval()
    routes_final=[]
    xv=get_batch(cfg.block_size+1)
    h=model.transformer.drop(model.transformer.wte(xv[:,:-1]))
    with torch.no_grad():
        for b in model.transformer.h:
            ln=b.ln1(h);_,cq=b.l3(ln)
            r=F.softmax(b.scheduler(cq.detach()),dim=-1)
            w=r.mean(dim=(0,1));routes_final.append(w.tolist())
            choice=w.argmax().item()
            if choice==0: out=b.l1(ln)
            elif choice==1: out,_=b.l2(ln)
            else: out,_=b.l3(ln)
            h=h+out+b.mlp(b.ln2(h))
    for l,(r1,r2,r3) in enumerate(routes_final):
        label=['L1','L2','L3'][max(enumerate([r1,r2,r3]),key=lambda x:x[1])[0]]
        print(f'  Layer {l:2d}: L1={r1:.3f} L2={r2:.3f} L3={r3:.3f} <- {label}')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--bsz',type=int,default=4);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-rina-sh')
    args=p.parse_args()
    cfg=RINA_SH_Config(vocab_size=50257,block_size=args.seq,use_int4=True,sparse_k=8,sparse_local_w=4)
    train(cfg,steps=args.steps,lr=args.lr,bsz=args.bsz,out=args.out)
