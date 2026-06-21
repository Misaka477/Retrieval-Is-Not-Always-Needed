"""纯 CE 续训: warmup + cosine 重启"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_a import RINA_A, RINA_A_Config

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device='cuda'

def train(cfg, warmup=5000, decay=200000, peak_lr=3e-5, bsz=2, seq=512, out='models/out-0.1b-ce'):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/mix_pretrain_llama.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10
    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_A(cfg).to(device)
    ckpt=torch.load(f'{out}/ce_final.pt',map_location=device,weights_only=False)['model']
    model.load_state_dict(ckpt,strict=False)
    print(f'Loaded ce_final.pt ({len(ckpt)} keys)')
    model.train()

    opt=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.dim()>=2],'weight_decay':0.01},
        {'params':[p for n,p in model.named_parameters() if p.dim()<2],'weight_decay':0.0},
    ],lr=0,betas=(0.9,0.95))
    total_steps=warmup+decay

    pbar=tqdm(range(total_steps));t0=time.time()
    log_file=open(f'{out}/train_r2.log','w',buffering=1)
    log_file.write('step\tce\tval\tlr\n')

    for step in pbar:
        # warmup + cosine scheduler
        if step<warmup:
            lr=peak_lr*step/warmup
        else:
            cos_step=step-warmup
            lr=peak_lr*0.5*(1+math.cos(math.pi*cos_step/decay))
        for pg in opt.param_groups: pg['lr']=lr

        x=get_batch(seq+1)
        l,ce,_=model(x[:,:-1],x[:,1:])
        opt.zero_grad();ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step%1000==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(seq+1)
                lv,val_ce,_=model(xv[:,:-1],xv[:,1:])
            model.train()
            msg=f'step {step+300000}: ce={ce.item():.2f} val={val_ce.item():.2f} lr={lr:.1e}'
            pbar.set_postfix_str(msg)
            log_file.write(msg+'\n')
            torch.save({'model':model.state_dict()},f'{out}/ce_r2_{step+300000}.pt')

    torch.save({'model':model.state_dict()},f'{out}/ce_r2_final.pt')
    log_file.close()
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--warmup',type=int,default=5000);p.add_argument('--decay',type=int,default=200000)
    p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--bsz',type=int,default=2)
    p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-0.1b-ce')
    args=p.parse_args()
    cfg=RINA_A_Config(vocab_size=128256,block_size=args.seq,use_int4=True,n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160)
    train(cfg,warmup=args.warmup,decay=args.decay,peak_lr=args.lr,bsz=args.bsz,seq=args.seq,out=args.out)
