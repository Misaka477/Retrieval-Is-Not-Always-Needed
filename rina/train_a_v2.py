"""正确的 0.1B 训练: CE + triplet margin 对比 loss，从 ce_final 续训"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_a import RINA_A, RINA_A_Config

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device='cuda'

def train(cfg, steps=100000, lr=1e-4, bsz=2, seq=512, out='models/out-0.1b-a-v2', resume=None):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/mix_pretrain_llama.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10
    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_A(cfg).to(device)
    offset=300000
    if resume:
        ckpt=torch.load(resume,map_location=device,weights_only=False)['model']
        model.load_state_dict(ckpt,strict=False)
        base=int(os.path.splitext(os.path.basename(resume))[0].split('_')[-1])
        offset=base
        steps=max(0,steps-(offset-300000))
        print(f'Resumed from {resume} (step {base}), remaining steps={steps}')
    else:
        ckpt=torch.load('models/out-0.1b-ce/ce_final.pt',map_location=device,weights_only=False)['model']
        model.load_state_dict(ckpt,strict=False)
        print(f'Loaded ce_final ({len(ckpt)} keys)')
    model.train()

    opt=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.dim()>=2],'weight_decay':0.01,'initial_lr':lr},
        {'params':[p for n,p in model.named_parameters() if p.dim()<2],'weight_decay':0.0,'initial_lr':lr},
    ],lr=lr,betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    pbar=tqdm(range(steps));t0=time.time()
    log_file=open(f'{out}/train.log','a',buffering=1)
    if not resume:
        log_file.write('step\tce\tcl\tval\tlr\n')

    for step in pbar:
        x=get_batch(seq+1)
        l,ce,lats=model(x[:,:-1],x[:,1:])
        cl=model.compute_contrastive_loss(lats)*1.0
        loss=ce+cl
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()
        if step%1000==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(seq+1)
                lv,val_ce,_=model(xv[:,:-1],xv[:,1:])
            model.train()
            msg=f'step {step+offset}: ce={ce.item():.2f} cl={cl.item():.4f} val={val_ce.item():.2f} lr={sched.get_last_lr()[0]:.1e}'
            pbar.set_postfix_str(msg)
            log_file.write(msg+'\n')
            torch.save({'model':model.state_dict()},f'{out}/a_{step+offset}.pt')

    torch.save({'model':model.state_dict()},f'{out}/a_final.pt')
    log_file.close()
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--bsz',type=int,default=2);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-0.1b-a-v2')
    p.add_argument('--resume',type=str,default=None,help='resume from checkpoint path')
    args=p.parse_args()
    cfg=RINA_A_Config(vocab_size=128256,block_size=args.seq,use_int4=True,n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160)
    train(cfg,steps=args.steps,lr=args.lr,bsz=args.bsz,seq=args.seq,out=args.out,resume=args.resume)
