"""Simple RL + live visualization, with wall death + distance gradient."""
import os, sys, random, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
DM=64; CELL_WALL,CELL_PATH,CELL_EXIT=0,1,2
DIRS=[(0,-1),(1,0),(0,1),(-1,0)]
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_DIM=14

def gen_maze(h=9,w=9):
    m=np.ones((h,w),np.int32)*CELL_WALL
    m[1,1]=CELL_PATH; st=[(1,1)]; rng=random.Random(42)
    while st:
        cy,cx=st[-1]; nb=[]
        for dy,dx in [(-2,0),(2,0),(0,-2),(0,2)]:
            ny,nx=cy+dy,cx+dx
            if 0<ny<h-1 and 0<nx<w-1 and m[ny,nx]==CELL_WALL: nb.append((ny,nx))
        if nb: ny,nx=rng.choice(nb)
        else: st.pop(); continue
        m[cy+(ny-cy)//2][cx+(nx-cx)//2]=CELL_PATH; m[ny,nx]=CELL_PATH; st.append((ny,nx))
    m[h-2,w-2]=CELL_EXIT
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        if m[h-2+dy,w-2+dx]==CELL_WALL: m[h-2+dy,w-2+dx]=CELL_PATH
    return m

def get_obs(m,p,f):
    o=[]
    for i in range(4):
        dy,dx=DIRS[(f+i)%4]; ny,nx=p[0]+dy,p[1]+dx
        c=m[ny,nx] if 0<=ny<m.shape[0] and 0<=nx<m.shape[1] else CELL_WALL
        o+=[0,0,0]; o[c]=1
    h,w=m.shape; o+=[p[0]/h,p[1]/w]
    return np.array(o)

def v_acts(m,p,f):
    a=[0,1]; dy,dx=DIRS[f]; ny,nx=p[0]+dy,p[1]+dx
    if 0<=ny<m.shape[0] and 0<=nx<m.shape[1] and m[ny,nx]!=CELL_WALL: a.append(2)
    return a

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(OBS_DIM,DM),nn.ReLU(),nn.Linear(DM,DM),nn.ReLU())
        self.policy=nn.Linear(DM,3); self.value=nn.Linear(DM,1)
    def forward(self,obs):
        h=self.net(obs); return self.policy(h),self.value(h)

model=Policy().to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=2e-4); print(f'{sum(p.numel() for p in model.parameters())/1e3:.1f}K')

ALL=[gen_maze() for _ in range(30)]
DEMO=gen_maze(9,9)
plt.ion()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
fig.suptitle('RL Maze - Watch it learn',fontsize=12,fontweight='bold')
cmap=ListedColormap(['#2d2d2d','#f0f0f0','#4ecdc4'])
rh=[]; t0=time.time()

for ep in range(30000):
    m=random.choice(ALL); h,w=m.shape; EX=(h-2,w-2)
    p=(random.randrange(0,h),random.randrange(0,w))
    while m[p]==CELL_WALL: p=(random.randrange(0,h),random.randrange(0,w))
    f=random.randint(0,3)
    lp_log=[]; v_log=[]; r_log=[]
    
    for _ in range(100):
        ot=torch.from_numpy(get_obs(m,p,f)).float().unsqueeze(0).to(DEVICE)
        va=v_acts(m,p,f); old_d=abs(p[0]-EX[0])+abs(p[1]-EX[1])
        logits,v_=model(ot); mask=torch.full((3,),-1e9,device=DEVICE)
        for a_ in va: mask[a_]=0
        probs=F.softmax(logits+mask,dim=-1)
        a=torch.multinomial(probs,1).item()
        lp=F.log_softmax(logits+mask,dim=-1)[0,a]
        
        if a==2:
            dy,dx=DIRS[f]; ny,nx=p[0]+dy,p[1]+dx
            if 0<=ny<h and 0<=nx<w and m[ny,nx]!=CELL_WALL: p=(ny,nx)
            else: r=-50; lp_log.append(lp); v_log.append(v_); r_log.append(r); break
        elif a==0: f=(f-1)%4
        else: f=(f+1)%4
        
        nd=abs(p[0]-EX[0])+abs(p[1]-EX[1])
        r=100 if m[p]==CELL_EXIT else (old_d-nd)*5-0.2
        lp_log.append(lp); v_log.append(v_); r_log.append(r)
        if m[p]==CELL_EXIT: break
    
    G=0; rets=[]
    for rv in reversed(r_log): G=rv+0.99*G; rets.insert(0,G)
    rets=torch.tensor(rets,device=DEVICE)
    vt=torch.cat(v_log).squeeze()
    loss=-(torch.stack(lp_log)*(rets-vt.detach())).mean()+0.5*F.mse_loss(vt,rets)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    rh.append(sum(r_log))
    
    # Live demo: show agent navigating step by step every episode
    avg_r=np.mean(rh[-100:]) if rh else 0
    ax2.clear()
    if len(rh)>50: c=np.convolve(rh,np.ones(50)/50,mode='valid'); ax2.plot(c,linewidth=1,color='#0984e3')
    ax2.set_title(f'Reward (last 100: {avg_r:.1f})'); ax2.grid(True,alpha=0.2)
    
    pp,pf=(1,1),0; ppath=[(1,1)]; last_pos=None; stuck=0
    model.eval()
    with torch.no_grad():
        for _ in range(25):
            if last_pos==pp: stuck+=1
            else: stuck=0
            last_pos=pp
            va=v_acts(DEMO,pp,pf)
            if stuck>=3: a=random.choice(va); stuck=0
            else:
                ot=torch.from_numpy(get_obs(DEMO,pp,pf)).float().unsqueeze(0).to(DEVICE)
                l,_=model(ot); mk=torch.full((3,),-1e9,device=DEVICE)
                for v__ in va: mk[v__]=0
                a=(l+mk).argmax().item()
            if a==2:
                dy,dx=DIRS[pf]; ny,nx=pp[0]+dy,pp[1]+dx
                if 0<=ny<DEMO.shape[0] and 0<=nx<DEMO.shape[1] and DEMO[ny,nx]!=CELL_WALL: pp=(ny,nx)
            elif a==0: pf=(pf-1)%4
            else: pf=(pf+1)%4
            ppath.append(pp)
            ax1.clear()
            ax1.imshow(DEMO,cmap=cmap,vmin=0,vmax=2,interpolation='nearest')
            pa=np.array(ppath); ax1.plot(pa[:,1],pa[:,0],'-',color='#74b9ff',alpha=0.4,linewidth=1.5)
            ax1.plot(pp[1],pp[0],'o',markersize=10,color='#ff6b6b',zorder=5)
            ex_=np.where(DEMO==CELL_EXIT)
            if len(ex_[0]): ax1.plot(ex_[1][0],ex_[0][0],'*',markersize=16,color='#ffd93d',zorder=4)
            ax1.set_title(f'Ep {ep} | {"EXIT!" if DEMO[pp]==CELL_EXIT else f"Steps {len(ppath)}"}')
            ax2.set_title(f'Reward (last 100: {avg_r:.1f})')
            plt.pause(0.02)
            if DEMO[pp]==CELL_EXIT: break
    model.train()

plt.ioff(); print(f'Done. {(time.time()-t0)/60:.1f}min'); plt.show()
