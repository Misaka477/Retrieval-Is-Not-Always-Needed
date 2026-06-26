"""纯 CE 续训: 从 checkpoint 继续"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_a import RINA_A, RINA_A_Config

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device='cuda'

def train(cfg, start_step=105000, total_steps=200000, lr=3e-5, bsz=2, seq=512, out='models/out-0.1b-ce'):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/mix_pretrain_llama.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10
    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_A(cfg).to(device)
    ckpt=torch.load(f'{out}/ce_{start_step}.pt',map_location=device,weights_only=False)['model']
    model.load_state_dict(ckpt,strict=False)
    print(f'Loaded step {start_step} ({len(ckpt)} keys)')
    model.train()

    opt=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.dim()>=2],'weight_decay':0.01,'initial_lr':lr},
        {'params':[p for n,p in model.named_parameters() if p.dim()<2],'weight_decay':0.0,'initial_lr':lr},
    ],lr=lr,betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total_steps,last_epoch=start_step-100000-1)
    pbar=tqdm(range(start_step-100000,total_steps));t0=time.time()
    log_file=open(f'{out}/train.log','a',buffering=1)

    for step in pbar:
        step_abs=step+100000
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
            log_msg=f'step {step_abs}: ce={ce.item():.2f} val={val_ce.item():.2f} lr={sched.get_last_lr()[0]:.1e}'
            print(log_msg,flush=True)
            log_file.write(log_msg+'\n')
            torch.save({'model':model.state_dict()},f'{out}/ce_{step_abs}.pt')

    torch.save({'model':model.state_dict()},f'{out}/ce_final.pt')
    log_file.close()
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=200000);p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--start',type=int,default=105000)
    p.add_argument('--bsz',type=int,default=2);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-0.1b-ce')
    args=p.parse_args()
    cfg=RINA_A_Config(vocab_size=128256,block_size=args.seq,use_int4=True,n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160)
    train(cfg,start_step=args.start,total_steps=args.steps,lr=args.lr,bsz=args.bsz,seq=args.seq,out=args.out)
