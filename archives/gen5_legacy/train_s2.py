"""Scheduler Phase 2: 解冻 backbone，调度头 + 三路联合微调"""
import os, math, time, argparse
import torch, numpy as np
from tqdm import tqdm
from rina.model_s import RINA_S, RINA_S_Config

device='cuda' if torch.cuda.is_available() else 'cpu'

def train(cfg, s_ckpt='models/out-rina-s/rina_s_final.pt',
          data_path='data/english_pretrain.npy', lr=1e-4, bsz=4, steps=5000, out='models/out-rina-s2'):
    os.makedirs(out,exist_ok=True)
    if not os.path.isabs(data_path):
        data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),data_path)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10

    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_S(cfg).to(device)

    # 加载调度层 checkpoint
    sd=torch.load(s_ckpt,map_location='cpu',weights_only=False)['model']
    own=dict(model.named_parameters())
    loaded=0
    for k,v in sd.items():
        if k in own and own[k].shape==v.shape:
            own[k].data.copy_(v);loaded+=1
    print(f'Loaded {loaded} keys from {s_ckpt}')

    # 解冻全部
    for p in model.parameters(): p.requires_grad=True
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable: {trainable} params (all)')

    # 调度头用更高 lr，backbone 用小 lr
    backbone_params=[p for n,p in model.named_parameters() if 'scheduler' not in n]
    scheduler_params=[p for n,p in model.named_parameters() if 'scheduler' in n]
    opt=torch.optim.AdamW([
        {'params':backbone_params,'lr':lr,'weight_decay':0.01},
        {'params':scheduler_params,'lr':lr*3,'weight_decay':0.0},
    ],betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    pbar=tqdm(range(steps));t0=time.time()

    for step in pbar:
        x=get_batch(cfg.block_size+1)
        l,loss,_=model(x[:,:-1],x[:,1:])
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()

        if step%500==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(cfg.block_size+1)
                lv,vloss,routes=model(xv[:,:-1],xv[:,1:])
                ce_v=vloss.item()
                assign=routes.argmax(-1).float().mean((0,1)).tolist()
            model.train()
            pbar.set_postfix(ce=f'{loss.item():.2f}',val=f'{ce_v:.2f}',
                             l1=f'{assign[0]:.2f}',l2=f'{assign[1]:.2f}',l3=f'{assign[2]:.2f}')
            torch.save({'model':model.state_dict(),'step':step,'routes':assign},
                       f'{out}/rina_s2_{step}.pt')

    torch.save({'model':model.state_dict()},f'{out}/rina_s2_final.pt')
    # 最终路由分析
    model.eval()
    with torch.no_grad():
        xv=get_batch(cfg.block_size+1)
        _,_,routes=model(xv[:,:-1],xv[:,1:])
        for l in range(cfg.n_layer):
            r=routes[0,:,l].mean(0)
            label=['L1','L2','L3'][r.argmax().item()]
            print(f'  Layer {l:2d}: L1={r[0]:.3f} L2={r[1]:.3f} L3={r[2]:.3f} <- {label}')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--bsz',type=int,default=4);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-rina-s2')
    p.add_argument('--ckpt',type=str,default='models/out-rina-s/rina_s_final.pt')
    args=p.parse_args()
    cfg=RINA_S_Config(vocab_size=50257,block_size=args.seq,use_int4=True,sparse_k=8,sparse_local_w=4)
    train(cfg,s_ckpt=args.ckpt,lr=args.lr,bsz=args.bsz,steps=args.steps,out=args.out)
