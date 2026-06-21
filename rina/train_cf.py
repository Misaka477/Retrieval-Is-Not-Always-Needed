"""Train CF: 置信度调度 — 训练置信度头 + 加载 backbone"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_cf import RINA_CF, RINA_CF_Config

device='cuda' if torch.cuda.is_available() else 'cpu'

def train(cfg, steps=3000, lr=1e-4, bsz=4, out='models/out-rina-cf'):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/english_pretrain.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10

    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_CF(cfg).to(device)
    own=dict(model.named_parameters())

    # 加载三路权重（L1/L3 只加载路径专用权重，L2 加载全部共享权重）
    def load_path(ckpt, src_sub, dst_sub, path_only=False):
        """加载权重, path_only=True 时只加载 dst_sub 路径的权重"""
        sd=torch.load(ckpt,map_location='cpu',weights_only=False)
        state=sd['model'] if 'model' in sd else sd
        own=dict(model.named_parameters())
        n=0
        for k,v in state.items():
            target=k.replace(src_sub,dst_sub)
            if not path_only or dst_sub in target:  # path_only 时只加载目标路径
                if target in own and own[target].shape==v.shape:
                    own[target].data.copy_(v);n+=1
        return n
    
    print('Loading backbone weights...')
    n1=load_path('models/out-rina-c/rina-gen5-route-c.pt','inertia.','l1.',path_only=True)
    n2=load_path('models/out-quant/rina-gen5-baseline-int4.pt','attn.','l2.')
    n3=load_path('models/out-rina-a-v3/rina-gen5-route-a-v3.pt','attn.','l3.',path_only=True)

    # 冻结全部, 只训置信度头
    for n,p in model.named_parameters():
        p.requires_grad=False
        if 'conf_head' in n: p.requires_grad=True
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable: {trainable} params (confidence heads only)')

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0.0)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    pbar=tqdm(range(steps));t0=time.time()

    for step in pbar:
        x=get_batch(cfg.block_size+1)
        l,ce,conf_loss,confs=model(x[:,:-1],x[:,1:])
        opt.zero_grad()
        conf_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        opt.step();sched.step()

        if step%500==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(cfg.block_size+1)
                lv,ce_v,_,_=model(xv[:,:-1],xv[:,1:])
                p=F.softmax(lv,dim=-1)
                real_ent=-(p*p.clamp(1e-8).log()).sum(-1)  # 原始熵
            model.train()
            pbar.set_postfix(ce=f'{ce.item():.2f}',val=f'{ce_v.item():.2f}',
                             conf=f'{confs.mean().item():.2f}',real=f'{real_ent.mean().item():.2f}')
            torch.save({'model':model.state_dict(),'step':step},
                       f'{out}/rina_cf_{step}.pt')

    torch.save({'model':model.state_dict()},f'{out}/rina_cf_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--bsz',type=int,default=4);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-rina-cf')
    args=p.parse_args()
    cfg=RINA_CF_Config(vocab_size=50257,block_size=args.seq,use_int4=True,sparse_k=8,sparse_local_w=4)
    train(cfg,steps=args.steps,lr=args.lr,bsz=args.bsz,out=args.out)
