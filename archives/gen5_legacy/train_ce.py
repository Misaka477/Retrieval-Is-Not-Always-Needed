"""纯 CE 续训: 从蒸馏 final 继续，100K 步"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_a import RINA_A, RINA_A_Config

device='cuda'

def train(cfg, total_steps=100000, lr=3e-5, bsz=2, seq=512, out='models/out-0.1b-ce'):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/mix_pretrain_llama.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10
    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_A(cfg).to(device)
    ckpt=torch.load('models/out-0.1b-distil/distil_final.pt',map_location=device,weights_only=False)['model']
    model.load_state_dict(ckpt,strict=False)
    print(f'Loaded distilled model ({len(ckpt)} keys)')
    model.train()
    log_file=open(f'{out}/train.log','w',buffering=1)

    opt=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.dim()>=2],'weight_decay':0.01,'initial_lr':lr},
        {'params':[p for n,p in model.named_parameters() if p.dim()<2],'weight_decay':0.0,'initial_lr':lr},
    ],lr=lr,betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total_steps)
    pbar=tqdm(range(total_steps));t0=time.time()
    log_file=open(f'{out}/train.log','w',buffering=1)
    log_file.write(f'step\tce\tval\tlr\n')

    for step in pbar:
        x=get_batch(seq+1)
        l,ce,_=model(x[:,:-1],x[:,1:])
        opt.zero_grad();ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()
        if step%1000==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(seq+1)
                lv,val_ce,_=model(xv[:,:-1],xv[:,1:])
            model.train()
            log_msg=f'step {step+100000}: ce={ce.item():.2f} val={val_ce.item():.2f} lr={sched.get_last_lr()[0]:.1e}'
            pbar.set_postfix_str(log_msg)
            log_file.write(log_msg+'\n')
            torch.save({'model':model.state_dict(),'step':step+100000},f'{out}/ce_{step+100000}.pt')

    torch.save({'model':model.state_dict()},f'{out}/ce_final.pt')
    log_file.close()
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--bsz',type=int,default=2);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-0.1b-ce')
    args=p.parse_args()
    cfg=RINA_A_Config(vocab_size=128256,block_size=args.seq,use_int4=True,n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160)
    train(cfg,total_steps=args.steps,lr=args.lr,bsz=args.bsz,seq=args.seq,out=args.out)
