"""实况RL训练：每轮结束都看你走，每一步都显示。"""
import os, sys, random, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DM = 64
CELL_WALL, CELL_PATH, CELL_EXIT = 0, 1, 2
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_DIM = 14

def generate_maze(h=11, w=11):
    maze = np.ones((h,w), dtype=np.int32)*CELL_WALL
    start = (1,1); maze[start]=CELL_PATH; stack=[start]; rng=random.Random(42)
    while stack:
        cy,cx=stack[-1]; nbs=[]
        for dy,dx in [(-2,0),(2,0),(0,-2),(0,2)]:
            ny,nx=cy+dy,cx+dx
            if 0<ny<h-1 and 0<nx<w-1 and maze[ny,nx]==CELL_WALL: nbs.append((ny,nx))
        if nbs: ny,nx=rng.choice(nbs)
        else: stack.pop(); continue
        maze[cy+(ny-cy)//2][cx+(nx-cx)//2]=CELL_PATH
        maze[ny,nx]=CELL_PATH; stack.append((ny,nx))
    maze[h-2,w-2]=CELL_EXIT
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        if maze[h-2+dy,w-2+dx]==CELL_WALL: maze[h-2+dy,w-2+dx]=CELL_PATH
    return maze

def get_obs(maze, pos, facing):
    obs=[]
    for i in range(4):
        dy,dx=DIRS[(facing+i)%4]; ny,nx=pos[0]+dy,pos[1]+dx
        cell=maze[ny,nx] if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] else CELL_WALL
        o=[0,0,0]; o[cell]=1; obs.extend(o)
    h,w=maze.shape; obs.append(pos[0]/h); obs.append(pos[1]/w)
    return np.array(obs)

def av_acts(maze, pos, facing):
    acts=[0,1]
    dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL: acts.append(2)
    return acts

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, DM), nn.ReLU(),
            nn.Linear(DM, DM), nn.ReLU(),
        )
        self.policy = nn.Linear(DM, 3)
        self.value = nn.Linear(DM, 1)
    def forward(self, obs):
        h = self.net(obs)
        return self.policy(h), self.value(h)
    def act(self, obs, valid_acts, eps=0.0):
        logits, v = self.forward(obs)
        mask = torch.full((3,), -1e9, device=DEVICE)
        for a in valid_acts: mask[a] = 0
        logits = logits + mask
        if random.random() < eps:
            a = random.choice(valid_acts)
        else:
            probs = F.softmax(logits, dim=-1)
            a = torch.multinomial(probs, 1).item()
        return a, F.log_softmax(logits, dim=-1)[0,a], v

model = Policy().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

DEMO = generate_maze(13, 13)
DEMO_EXIT = (11, 11)
ALL_MAZES = [generate_maze() for _ in range(30)]

plt.ion()
fig, (ax_maze, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('RL training - watching agent learn to walk', fontsize=13, fontweight='bold')
cmap = ListedColormap(['#2d2d2d', '#f0f0f0', '#4ecdc4'])

reward_hist = []
upd = 200
t0 = time.time()

for ep in range(50000):
    maze = random.choice(ALL_MAZES)
    h,w = maze.shape; EXIT = (h-2, w-2)
    pos = (random.randrange(0,h), random.randrange(0,w))
    while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
    facing = random.randint(0,3)
    
    lp_log=[]; v_log=[]; r_log=[]
    for _ in range(150):
        ot = torch.from_numpy(get_obs(maze,pos,facing)).float().unsqueeze(0).to(DEVICE)
        valid = av_acts(maze,pos,facing)
        old_d = abs(pos[0]-EXIT[0])+abs(pos[1]-EXIT[1])
        a,lp,v = model.act(ot, valid, eps=max(0.02,0.4*(1-ep/8000)))
        if a==2:
            dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
            if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL: pos=(ny,nx)
            else: r=-50; lp_log.append(lp); v_log.append(v); r_log.append(r); break  # 撞墙死
        elif a==0: facing=(facing-1)%4
        else: facing=(facing+1)%4
        dist_progress = old_d-(abs(pos[0]-EXIT[0])+abs(pos[1]-EXIT[1]))
        r = 100 if maze[pos]==CELL_EXIT else dist_progress*5 - 0.2  # 靠近+5，每步只烧0.2
        lp_log.append(lp); v_log.append(v); r_log.append(r)
        if maze[pos]==CELL_EXIT: break
    
    G=0; rets=[]
    for r in reversed(r_log):
        G = r+0.99*G; rets.insert(0,G)
    rets=torch.tensor(rets,device=DEVICE)
    vt=torch.cat(v_log).squeeze()
    loss = -(torch.stack(lp_log)*(rets-vt.detach())).mean() + 0.5*F.mse_loss(vt,rets)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    reward_hist.append(sum(r_log))
    
    if ep % upd == 0:
        model.eval()
        # Walk the demo maze step by step, show each move
        ppos, pfacing = (1,1), 0
        ppath = [(1,1)]
        ax_maze.clear()
        ax_maze.imshow(DEMO, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        ax_maze.set_xticks([]); ax_maze.set_yticks([])
        
        for s in range(100):
            ot = torch.from_numpy(get_obs(DEMO,ppos,pfacing)).float().unsqueeze(0).to(DEVICE)
            valid = av_acts(DEMO,ppos,pfacing)
            a,_,_ = model.act(ot, valid)
            if a==2:
                dy,dx=DIRS[pfacing]; ny,nx=ppos[0]+dy,ppos[1]+dx
                if 0<=ny<DEMO.shape[0] and 0<=nx<DEMO.shape[1] and DEMO[ny,nx]!=CELL_WALL: ppos=(ny,nx)
            elif a==0: pfacing=(pfacing-1)%4
            else: pfacing=(pfacing+1)%4
            ppath.append(ppos)
            
            # Update maze view
            ax_maze.clear()
            ax_maze.imshow(DEMO, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
            pa = np.array(ppath)
            ax_maze.plot(pa[:,1], pa[:,0], '-', color='#74b9ff', alpha=0.4, linewidth=1.5)
            ax_maze.plot(ppos[1], ppos[0], 'o', markersize=10, color='#ff6b6b', zorder=5)
            if a==2:
                dy,dx=DIRS[pfacing]
                ax_maze.arrow(ppos[1], ppos[0], dx*0.3, dy*0.3, head_width=0.3,
                             head_length=0.3, fc='#ff6b6b', ec='#ff6b6b', zorder=6)
            ex_ = np.where(DEMO==CELL_EXIT)
            if len(ex_[0]): ax_maze.plot(ex_[1][0], ex_[0][0], '*', markersize=16, color='#ffd93d', zorder=4)
            
            avg = np.mean(reward_hist[-200:]) if len(reward_hist)>=200 else sum(reward_hist)/max(len(reward_hist),1)
            reached = DEMO[ppos]==CELL_EXIT
            ax_maze.set_title(f'Ep {ep} | AvgRew {avg:.1f} | Steps {len(ppath)}', fontsize=10)
            
            # Right side: training curve
            ax_plot.clear()
            if len(reward_hist)>50:
                c = np.convolve(reward_hist, np.ones(50)/50, mode='valid')
                ax_plot.plot(c, linewidth=1, color='#0984e3')
            ax_plot.set_title(f'Training reward ({len(reward_hist)} eps)')
            ax_plot.set_xlabel('Episode'); ax_plot.grid(True, alpha=0.2)
            ax_plot.axhline(y=95, color='#ff6b6b', linestyle='--', alpha=0.5, label='exit=100')
            
            plt.pause(0.05)
            if reached: break
        
        model.train()

plt.ioff()
print(f'Done. {time.time()-t0:.0f}s')
plt.show()
