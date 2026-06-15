"""Watch GRU+position navigate a maze, step by step.
"""
import os, sys, random, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DM = 64
CELL_WALL, CELL_PATH, CELL_EXIT = 0, 1, 2
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_maze(h=13, w=13):
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
    return np.array(obs,dtype=np.float32)

def av_acts(maze, pos, facing):
    acts=[0,1]
    dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL: acts.append(2)
    return acts

class PosGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder=nn.Linear(12,DM)
        self.act_embed=nn.Embedding(3,DM//2)
        self.cell=nn.GRUCell(DM+DM//2,DM)
        self.obs_predictor=nn.Linear(DM,12)
        self.pos_predictor=nn.Linear(DM,2)
    def forward(self,obs,act,h=None):
        oe=self.obs_encoder(obs); ae=self.act_embed(act)
        inp=torch.cat([oe,ae.squeeze(1)],-1)
        if h is None: h=torch.zeros(obs.size(0),DM,device=DEVICE)
        h=self.cell(inp,h)
        return self.obs_predictor(h), self.pos_predictor(h), h

model = PosGRU().to(DEVICE)
ckpt = torch.load(os.path.join(BASE_DIR, 'checkpoints', 'maze_position.pt'), weights_only=False, map_location=DEVICE)
model.load_state_dict(ckpt['model']); model.eval()
print('Loaded GRU+position')

maze = generate_maze(13, 13)
h, w = maze.shape
pos, facing = (1, 1), 0
path = [pos]
model_state = None

plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
cmap = ListedColormap(['#2d2d2d', '#f0f0f0', '#4ecdc4'])

for step in range(200):
    acts = av_acts(maze, pos, facing)
    
    obs = get_obs(maze, pos, facing)
    ot = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
    
    best_act, best_score = acts[0], -1e9
    EXIT = (11, 11)  # bottom-right
    with torch.no_grad():
        for a in acts:
            # Use model's position delta to predict where we'll end up
            _, pred_delta, _ = model(ot, torch.tensor([[a]], device=DEVICE), model_state)
            dy, dx = pred_delta[0].cpu().numpy()
            ny, nx = pos[0] + round(dy), pos[1] + round(dx)
            
            # Clip to maze boundaries
            ny, nx = max(0,min(h-1,ny)), max(0,min(w-1,nx))
            
            if maze[ny,nx] == CELL_EXIT:
                score = 1000  # big reward
            else:
                dist = abs(ny-EXIT[0]) + abs(nx-EXIT[1])
                score = -dist  # closer = better
            if score > best_score: best_act, best_score = a, score
    
    # Execute best action
    if best_act == 2:
        dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
        if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL: pos = (ny,nx)
    elif best_act == 0: facing = (facing-1)%4
    else: facing = (facing+1)%4
    
    path.append(pos)
    
    # Render
    ax.clear()
    ax.imshow(maze, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    pa = np.array(path)
    ax.plot(pa[:,1], pa[:,0], '-', color='#74b9ff', alpha=0.5, linewidth=2)
    ax.plot(pos[1], pos[0], 'o', markersize=14, color='#ff6b6b', zorder=5)
    dy2,dx2 = DIRS[facing]
    ax.arrow(pos[1], pos[0], dx2*0.3, dy2*0.3, head_width=0.3, head_length=0.3,
             fc='#ff6b6b', ec='#ff6b6b', zorder=6)
    ey,ex = np.where(maze==CELL_EXIT)
    if len(ey): ax.plot(ex[0], ey[0], '*', markersize=20, color='#ffd93d', zorder=4)
    
    dist = abs(pos[0]-11) + abs(pos[1]-11)
    ax.set_title(f'GRU+pos Step {step+1}  |  Dist: {dist}', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    
    plt.pause(0.15)
    if maze[pos] == CELL_EXIT: break

print(f'Done. {step+1} steps, dist={dist}')
plt.ioff(); plt.show()
