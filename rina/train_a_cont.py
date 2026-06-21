"""Continue training Route A 0.1B from final checkpoint"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_a import RINA_A, RINA_A_Config

device='cuda' if torch.cuda.is_available() else 'cpu'

def train(cfg, start_step=10000, add_steps=40000, total_steps=50000,
          lr=1e-4, bsz=4, out='models/out-0.1b-a'):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/mix_pretrain.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10

    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_A(cfg).to(device)
    ckpt=torch.load(f'{out}/rina_a_final.pt',map_location=device,weights_only=False)['model']
    model.load_state_dict(ckpt,strict=False)
    model.train()
    print(f'Loaded {len(ckpt)} keys from {out}/rina_a_final.pt')

    opt=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.dim()>=2],'weight_decay':0.01,'initial_lr':lr},
        {'params':[p for n,p in model.named_parameters() if p.dim()<2],'weight_decay':0.0,'initial_lr':lr},
    ],lr=lr,betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total_steps,last_epoch=start_step-1)
    pbar=tqdm(range(start_step,total_steps));t0=time.time()

    for step in pbar:
        x=get_batch(cfg.block_size+1)
        l,ce,lats=model(x[:,:-1],x[:,1:])
        cl=model.compute_contrastive_loss(lats)*1.0
        loss=ce+cl
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()

        if step%1000==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(cfg.block_size+1)
                lv,val_ce,_=model(xv[:,:-1],xv[:,1:])
            model.train()
            pbar.set_postfix(ce=f'{ce.item():.2f}',cl=f'{cl.item():.2f}',val=f'{val_ce.item():.2f}',lr=f'{sched.get_last_lr()[0]:.1e}')
            torch.save({'model':model.state_dict(),'step':step},f'{out}/rina_a_{step}.pt')

    torch.save({'model':model.state_dict()},f'{out}/rina_a_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=40000);p.add_argument('--start',type=int,default=10000)
    p.add_argument('--total',type=int,default=50000);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--bsz',type=int,default=4);p.add_argument('--out',type=str,default='models/out-0.1b-a')
    args=p.parse_args()
    cfg=RINA_A_Config(vocab_size=50257,block_size=512,use_int4=True,n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160)
    train(cfg,start_step=args.start,add_steps=args.steps,total_steps=args.total,lr=args.lr,bsz=args.bsz,out=args.out)
