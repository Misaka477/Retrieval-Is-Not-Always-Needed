"""Train CF: 全参联合训练 — L1+L2+L3+conf_head 端到端"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_cf import RINA_CF, RINA_CF_Config

device='cuda' if torch.cuda.is_available() else 'cpu'
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train(cfg, steps=3000, lr=1e-3, bsz=4, out='models/out-rina-cf', resume=None):
    data_path=os.path.join(BASE,'data/mix_pretrain_llama.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10
    log_file=open(os.path.join(out,'train.log'),'a' if resume else 'w')

    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_CF(cfg).to(device)

    start_step=0
    if resume:
        sd=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(sd['model'],strict=False)
        start_step=sd.get('step',0)
        print(f'Resumed from step {start_step}')
    else:
        def load_path(ckpt, src_sub, dst_sub, path_only=False):
            sd_ckpt=torch.load(ckpt,map_location='cpu',weights_only=False)
            state=sd_ckpt['model'] if 'model' in sd_ckpt else sd_ckpt
            own=dict(model.named_parameters())
            n=0
            for k,v in state.items():
                target=k.replace(src_sub,dst_sub)
                if not path_only or dst_sub in target:
                    if target in own and own[target].shape==v.shape:
                        own[target].data.copy_(v);n+=1
            return n
        print('Loading backbone weights...')
        n1=load_path(os.path.join(BASE,'models/out-0.1b-c-distil-multistep/c_final.pt'),'inertia.','l1.',path_only=True)
        n2=load_path(os.path.join(BASE,'models/out-0.1b-a-v2/a_final.pt'),'attn.','l2.')
        n3=load_path(os.path.join(BASE,'models/out-0.1b-a-v2/a_final.pt'),'attn.','l3.',path_only=True)
        print(f'  Loaded: L1={n1} L2={n2} L3={n3} weights')

    trainable=sum(p.numel() for p in model.parameters())
    print(f'Full joint training: {trainable} params (all paths unfrozen)')

    param_groups=[
        {'params':[p for n,p in model.named_parameters() if 'conf_head' in n],'lr':lr},
        {'params':[p for n,p in model.named_parameters() if 'conf_head' not in n],'lr':lr*0.1},
    ]
    opt=torch.optim.AdamW(param_groups,weight_decay=0.1,betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    pbar=tqdm(range(start_step,steps));t0=time.time()

    for step in pbar:
        x=get_batch(cfg.block_size+1)
        l,ce,conf_loss,confs=model(x[:,:-1],x[:,1:],return_conf=True,l1_k=-1)
        loss=ce+conf_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()

        if step%50==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(cfg.block_size+1)
                lv,ce_v,_,confs_eval=model(xv[:,:-1],xv[:,1:],return_conf=True,l1_k=-1)
                p=F.softmax(lv,dim=-1)
                real_ent=-(p*p.clamp(1e-8).log()).sum(-1)
            model.train()
            msg=f'step={step} ce={ce.item():.2f} val={ce_v.item():.2f} conf={confs.mean().item():.2f} real={real_ent.mean().item():.2f}'
            log_file.write(msg+'\n');log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.2f}',val=f'{ce_v.item():.2f}',
                             conf=f'{confs.mean().item():.2f}',real=f'{real_ent.mean().item():.2f}')
        if step%500==0:
            torch.save({'model':model.state_dict(),'step':step},f'{out}/rina_cf_{step}.pt')
            keep=sorted([f for f in os.listdir(out) if f.startswith('rina_cf_') and f.endswith('.pt')],
                       key=lambda x:int(x.split('_')[-1].split('.')[0]))
            for f in keep[:-4]: os.remove(os.path.join(out,f))

    log_file.close()
    torch.save({'model':model.state_dict()},f'{out}/rina_cf_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=50000);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--bsz',type=int,default=1);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-rina-cf')
    p.add_argument('--resume',type=str,default=None)
    args=p.parse_args()
    cfg=RINA_CF_Config(vocab_size=128256,block_size=args.seq,use_int4=True,
        n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160,head_dim=64,
        sparse_k=16,sparse_window=32,sparse_local_w=4,ssm_steps=3,
        confidence_thresholds=(3.0,7.0,9.0))
    train(cfg,steps=args.steps,lr=args.lr,bsz=args.bsz,out=args.out,resume=args.resume)
