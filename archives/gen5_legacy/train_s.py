"""Train Scheduler: 三路分别加载各自 checkpoint, 只训调度头"""
import os, math, time, argparse
import torch, numpy as np
from tqdm import tqdm
from rina.model_s import RINA_S, RINA_S_Config

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_path(model, ckpt_path, prefix):
    """从 checkpoint 加载权重到指定的路径前缀（如 transformer.h.0.l1）"""
    sd=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    state=sd['model'] if 'model' in sd else sd
    own=dict(model.named_parameters())
    loaded=0
    for k,v in state.items():
        # 映射: checkpoints 的 key → scheduler 的 prefix key
        # l2/l3 加载 MLA/MHA 层的 key（w_dqkv, w_uq, c_proj...）
        # l1 加载 InertiaLayer 的 key（w_dq, w_mem, w_decay, w_out...）
        target=f'{prefix}.{k}'
        if target in own and own[target].shape==v.shape:
            own[target].data.copy_(v);loaded+=1
    return loaded

def train(cfg, steps=3000, lr=1e-3, bsz=4, out='models/out-rina-s'):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/english_pretrain.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10

    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_S(cfg).to(device)
    own=dict(model.named_parameters())

    def load_into(ckpt_path, src_sub, dst_sub):
        """加载 checkpoint 中 src_sub 替换为 dst_sub 的 key"""
        sd=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        state=sd['model'] if 'model' in sd else sd
        n=0
        for k,v in state.items():
            target=k.replace(src_sub,dst_sub)
            if target in own and own[target].shape==v.shape:
                own[target].data.copy_(v);n+=1
        return n

    print('Loading L1 (Inertia Wave)...')
    n1=load_into('models/out-rina-c/rina-gen5-route-c.pt','inertia.','l1.')
    print(f'  L1: {n1} keys')

    print('Loading L2 (Full Attention, int4)...')
    n2=load_into('models/out-quant/rina-gen5-baseline-int4.pt','attn.','l2.')
    print(f'  L2: {n2} keys')

    print('Loading L3 (Sparse Index Attention)...')
    n3=load_into('models/out-rina-a-v3/rina-gen5-route-a-v3.pt','attn.','l3.')
    print(f'  L3: {n3} keys')

    total=n1+n2+n3
    own_keys=len(own)
    print(f'Total: {total}/{own_keys} keys loaded')

    # 冻结除调度头外的全部权重
    for n,p in model.named_parameters():
        p.requires_grad='scheduler' in n
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable: {trainable} params (scheduler heads only)')
    model.train()

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0.01)
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
                ent=-(routes*routes.clamp(1e-8).log()).sum(-1).mean().item()
                assign=routes.argmax(-1).float().mean((0,1)).tolist()
            model.train()
            pbar.set_postfix(ce=f'{loss.item():.2f}',val=f'{ce_v:.2f}',
                             ent=f'{ent:.2f}',l1=f'{assign[0]:.2f}',l2=f'{assign[1]:.2f}',l3=f'{assign[2]:.2f}')
            torch.save({'model':model.state_dict(),'step':step,'routes':assign},
                       f'{out}/rina_s_{step}.pt')

    torch.save({'model':model.state_dict()},f'{out}/rina_s_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--bsz',type=int,default=4);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-rina-s')
    args=p.parse_args()
    cfg=RINA_S_Config(vocab_size=50257,block_size=args.seq,use_int4=True,sparse_k=8,sparse_local_w=4)
    train(cfg,steps=args.steps,lr=args.lr,bsz=args.bsz,out=args.out)
