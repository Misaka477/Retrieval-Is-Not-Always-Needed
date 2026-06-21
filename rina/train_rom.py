"""Train ROM: 模型在不确定时从 ROM 查事实, 代替幻觉"""
import os, math, time, argparse
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_cf import RINA_CF, RINA_CF_Config

device='cuda' if torch.cuda.is_available() else 'cpu'

def train(cfg, steps=3000, lr=1e-4, bsz=4, out='models/out-rina-rom'):
    data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/english_pretrain.npy')
    os.makedirs(out,exist_ok=True)
    data=np.load(data_path,mmap_mode='r')
    N=len(data);val_len=N//10

    def get_batch(s):
        pos=np.random.randint(0,N-val_len-s-1,(bsz,))
        x=np.array([data[p:p+s] for p in pos])
        return torch.from_numpy(x).long().to(device)

    model=RINA_CF(cfg).to(device)
    sd=torch.load('models/out-rina-cf/rina_cf_final.pt',map_location='cpu',weights_only=False)['model']
    model.load_state_dict(sd,strict=False)
    for p in model.parameters(): p.requires_grad=True
    print(f'Trainable: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    opt=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if 'conf_head' not in n],'lr':lr,'weight_decay':0.01},
        {'params':[p for n,p in model.named_parameters() if 'conf_head' in n],'lr':lr*5,'weight_decay':0.0},
    ],betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    pbar=tqdm(range(steps));t0=time.time()

    for step in pbar:
        x=get_batch(cfg.block_size+1)
        B,T=x.shape

        # 一半样本是"知识"(ROM), 另一半是"查询"(target)
        # ROM 样本的前 32 token 的 latent → FAISS 索引
        # 查询样本做预测, 低置信度时检索并注入 ROM

        # 构建 ROM latent
        rom_idx=torch.arange(B//2)
        qry_idx=torch.arange(B//2,B)

        rom_cq=None
        with torch.no_grad():
            h_rom=model.transformer.drop(model.transformer.wte(x[rom_idx,:-1]))
            for li,blk in enumerate(model.transformer.h):
                ln=blk.ln1(h_rom)
                if li==5:
                    rom_cq=blk.l2.q_norm(blk.l2.w_dqkv(ln)).detach()
                o,_=blk.l2(ln)
                h_rom=h_rom+o+blk.mlp(blk.ln2(h_rom))

        # 查询样本 forward + ROM 注入
        x_in=x[qry_idx,:-1]
        x_tgt=x[qry_idx,1:]
        h=model.transformer.drop(model.transformer.wte(x_in))
        rom_used=0

        for li,blk in enumerate(model.transformer.h):
            ln=blk.ln1(h)
            kv=None
            if li>=4 and li<=7 and rom_cq is not None:
                # 对每个查询样本，检一次置信度
                cq=blk.l2.q_norm(blk.l2.w_dqkv(ln))
                for i in range(len(qry_idx)):
                    conf=model.transformer.h[li].conf_head(cq[i:i+1]).mean().item()
                    if conf>cfg.confidence_thresholds[2]:
                        # 注入 ROM: 用当前 c_q 检索 rom_cq 中最相关的 token
                        d=blk.l2
                        cq_q=cq[i:i+1,-1:]  # [1,1,d_c]
                        sim=torch.bmm(F.normalize(rom_cq,-1),F.normalize(cq_q,-1).permute(0,2,1))
                        _,rmk=torch.topk(sim.squeeze(-1),min(8,rom_cq.size(1)),dim=-1)
                        sel=rom_cq[i:i+1,rmk]  # [1,8,d_c]
                        k_rom=d.w_uk(sel).view(1,-1,d.n_kv,d.d_h).transpose(1,2)
                        v_rom=d.w_k2v(d.w_uk(sel)).view(1,-1,d.n_kv,d.d_h).transpose(1,2)
                        if d.n_rep>1:
                            k_rom=k_rom.repeat_interleave(d.n_rep,1)
                            v_rom=v_rom.repeat_interleave(d.n_rep,1)
                        kv=(k_rom,v_rom)
                        rom_used+=1
                        break  # 每层最多注入一个样本
            o,_=blk.l2(ln,kv)
            h=h+o+blk.mlp(blk.ln2(h))

        h=model.transformer.ln_f(h)
        l=model.lm_head(h)
        ce=F.cross_entropy(l.view(-1,l.size(-1)),x_tgt.reshape(-1),ignore_index=-1)

        opt.zero_grad();ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()

        if step%500==0:
            model.eval()
            with torch.no_grad():
                xv=get_batch(cfg.block_size+1)
                lv,val_ce,_,_=model(xv[:,:-1],xv[:,1:])
            model.train()
            pbar.set_postfix(ce=f'{ce.item():.2f}',val=f'{val_ce.item():.2f}',rom=f'{rom_used}')
            torch.save({'model':model.state_dict(),'step':step},f'{out}/rina_rom_{step}.pt')

    torch.save({'model':model.state_dict()},f'{out}/rina_rom_final.pt')
    print(f'Done in {(time.time()-t0)/60:.1f}min')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--bsz',type=int,default=4);p.add_argument('--seq',type=int,default=512)
    p.add_argument('--out',type=str,default='models/out-rina-rom')
    args=p.parse_args()
    cfg=RINA_CF_Config(vocab_size=50257,block_size=args.seq,use_int4=True)
    train(cfg,steps=args.steps,lr=args.lr,bsz=args.bsz,out=args.out)
