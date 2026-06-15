"""Load trained maze model and watch the agent navigate in real-time.
"""
import os, sys, random, time
import torch, torch.nn as nn
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

class MazeWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder=nn.Linear(12,DM)
        self.act_embed=nn.Embedding(3,DM//2)
        self.cell=nn.GRUCell(DM+DM//2,DM)
        self.obs_predictor=nn.Linear(DM,12)
    def forward(self,obs,act,h=None):
        oe=self.obs_encoder(obs); ae=self.act_embed(act)
        inp=torch.cat([oe,ae.squeeze(1)],-1)
        if h is None: h=torch.zeros(obs.size(0),DM,device=DEVICE)
        h=self.cell(inp,h)
        return self.obs_predictor(h),h

# Load
model = MazeWM().to(DEVICE)
ckpt = torch.load(os.path.join(BASE_DIR, 'checkpoints', 'maze_trained.pt'), weights_only=False, map_location=DEVICE)
model.load_state_dict(ckpt['model']); model.eval()
print(f'Loaded model (step {ckpt.get("step","?")}, loss={ckpt.get("loss","?"):.4f})')

# Generate maze
maze = generate_maze(15, 15)
h, w = maze.shape
START = (1, 1)
EXIT = (h-2, w-2)

print(f'Maze: {h}x{w}, start={START}, exit={EXIT}')
print('Close window to exit.\n')

@torch.no_grad()
def navigate(model, maze, start, max_steps=200):
    """Model-guided navigation."""
    pos, facing = start, 0
    path = [pos]
    visits = {pos: 0}
    h_state = None
    
    for s in range(max_steps):
        if maze[pos] == CELL_EXIT: break
        
        acts = av_acts(maze, pos, facing)
        best_act, best_score = acts[0], -999
        
        for a in acts:
            obs = get_obs(maze, pos, facing)
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
            at = torch.tensor([[a]], device=DEVICE)
            pred, _ = model(ot, at)
            pc = pred.squeeze(0).cpu().numpy().reshape(4, 3)
            
            score = pc[0, CELL_PATH]
            if a == 2:
                dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
                if 0<=ny<h and 0<=nx<w and maze[ny,nx]==CELL_EXIT: score = 10
            score -= 0.05 * visits.get(tuple(pos), 0)
            
            if score > best_score: best_score, best_act = score, a
        
        if best_act == 2:
            dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
            if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL: pos = (ny,nx)
        elif best_act == 0: facing = (facing-1)%4
        else: facing = (facing+1)%4
        
        path.append(pos)
        visits[pos] = visits.get(pos, 0) + 1
        
        # Render
        yield path, pos, facing, s+1, max_steps

# Live rendering
plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
cmap = ListedColormap(['#2d2d2d', '#f0f0f0', '#4ecdc4'])
trail_color = '#74b9ff'
agent_color = '#ff6b6b'

for path, pos, facing, step, total in navigate(model, maze, START):
    ax.clear()
    ax.imshow(maze, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    
    pa = np.array(path)
    ax.plot(pa[:,1], pa[:,0], '-', color=trail_color, alpha=0.6, linewidth=2)
    ax.plot(pos[1], pos[0], 'o', markersize=14, color=agent_color, zorder=5)
    
    dy,dx = DIRS[facing]
    ax.arrow(pos[1], pos[0], dx*0.3, dy*0.3, head_width=0.3, head_length=0.3,
             fc=agent_color, ec=agent_color, zorder=6)
    
    ey,ex = np.where(maze==CELL_EXIT)
    if len(ey): ax.plot(ex[0], ey[0], '*', markersize=22, color='#ffd93d', zorder=4)
    ax.plot(START[1], START[0], 's', markersize=8, color='white', zorder=3)
    
    status = '✅ EXIT!' if maze[pos]==CELL_EXIT else '🚶 Walking...'
    ax.set_title(f'Step {step}/{total}  |  {status}  |  Pos: {pos}', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    
    # POV
    dir_labels = ['↑','→','↓','←']
    obs = get_obs(maze, pos, facing)
    cell_chr = {CELL_WALL: '██', CELL_PATH: '··', CELL_EXIT: 'EE'}
    pov = '  '.join(f'{l}:{cell_chr[np.argmax(obs[i*3:(i+1)*3])]}' for i,l in enumerate(dir_labels))
    ax.text(0.5, -0.03, f'POV: {pov}', transform=ax.transAxes,
            fontsize=8, ha='center', family='monospace')
    
    plt.pause(0.08)

plt.ioff()
print(f'Done. Reached exit in {step} steps.')
plt.show()
