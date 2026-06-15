"""Interactive maze navigation with matplotlib window.
Opens a GUI window showing agent navigating + model's mental state.
"""
import os, sys, random, time, math
import torch, torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DM = 64
CELL_WALL, CELL_PATH, CELL_EXIT = 0, 1, 2
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Maze ──
def generate_maze(h=13, w=13):
    maze = np.ones((h,w), dtype=np.int32)*CELL_WALL
    start = (1,1); maze[start]=CELL_PATH; stack=[start]; rng=random.Random(42)
    while stack:
        cy,cx=stack[-1]; nbs=[]
        for dy,dx in [(-2,0),(2,0),(0,-2),(0,2)]:
            ny,nx=cy+dy,cx+dx
            if 0<ny<h-1 and 0<nx<w-1 and maze[ny,nx]==CELL_WALL: nbs.append((ny,nx))
        if nbs:
            ny,nx=rng.choice(nbs)
            maze[cy+(ny-cy)//2][cx+(nx-cx)//2]=CELL_PATH
            maze[ny,nx]=CELL_PATH; stack.append((ny,nx))
        else: stack.pop()
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

def get_valid(maze, pos, facing):
    acts=[0,1]
    dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL:
        acts.append(2)
    return acts

# ── Model ──
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

model=MazeWM().to(DEVICE)
ckpt=torch.load(os.path.join(BASE_DIR,'checkpoints','maze_wm.pt'),weights_only=False,map_location=DEVICE)
model.load_state_dict(ckpt['model']); model.eval()

# ── Setup ──
maze=generate_maze(13,13)
h,w=maze.shape
pos, facing = (1,1), random.randint(0,3)
path=[pos]
model_state = None  # mental state

# Colors
wall_color='#2d2d2d'
path_color='#f0f0f0'
exit_color='#4ecdc4'
agent_color='#ff6b6b'
trail_color='#74b9ff'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Maze World Model — Live Navigation', fontsize=14, fontweight='bold')

cmap = ListedColormap([wall_color, path_color, exit_color])

# All state encodings for visualization
all_states = {}  # (y,x) → hidden state vector

for step in range(200):
    # ── Agent decides action ──
    acts = get_valid(maze, pos, facing)
    # Wall-following: try forward, if blocked turn right, then left, then back
    if 2 in acts:
        act = 2
    else:
        # Rotate: try right first, then left, then back
        for turn_act in [1, 0]:  # right, left
            if turn_act in acts:
                act = turn_act
                break
        else:
            act = random.choice(acts)
    
    # ── Model prediction ──
    obs = get_obs(maze, pos, facing)
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
    act_t = torch.tensor([[act]], device=DEVICE)
    
    with torch.no_grad():
        pred_obs, model_state = model(obs_t, act_t, model_state if step > 0 else None)
    
    pred_cells = pred_obs.squeeze(0).cpu().numpy().reshape(4,3).argmax(axis=1)
    
    # Store state for this position
    all_states[(pos[0], pos[1], facing)] = model_state.squeeze(0).cpu().numpy()
    
    # ── Execute action ──
    dy, dx = DIRS[facing]
    if act == 2:
        ny, nx = pos[0]+dy, pos[1]+dx
        if maze[ny,nx] != CELL_WALL: pos = (ny, nx)
    elif act == 0: facing = (facing - 1) % 4
    else: facing = (facing + 1) % 4
    path.append(pos)
    
    # ── Render ──
    ax1.clear(); ax2.clear()
    
    # LEFT: maze top-down
    ax1.imshow(maze, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    
    # Trail
    if len(path) > 1:
        path_arr = np.array(path)
        ax1.plot(path_arr[:, 1], path_arr[:, 0], '-', color=trail_color, alpha=0.6, linewidth=2, zorder=3)
    
    # Agent
    ax1.plot(pos[1], pos[0], 'o', markersize=16, color=agent_color, zorder=5)
    dy2, dx2 = DIRS[facing]
    ax1.arrow(pos[1], pos[0], dx2*0.3, dy2*0.3, head_width=0.3, head_length=0.3,
              fc=agent_color, ec=agent_color, zorder=6)
    
    # Exit marker
    ey, ex = np.where(maze == CELL_EXIT)
    if len(ey): ax1.plot(ex[0], ey[0], '*', markersize=24, color=exit_color, zorder=4)
    
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f'Step {step+1}  |  Pos: {pos}  Dir: {["↑","→","↓","←"][facing]}', fontsize=11)
    
    # Prediction view
    dir_labels = ['↑ F', '→ R', '↓ B', '← L']
    true_cell = ['W', '·', 'E']
    pred_text = '  '.join(f'{l}={true_cell[p]}' for l,p in zip(dir_labels, pred_cells))
    ax1.text(0.5, -0.05, f'Pred={pred_text}',
             transform=ax1.transAxes, fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # RIGHT: state space (PCA of visited states)
    if len(all_states) >= 3:
        keys = list(all_states.keys())
        vecs = np.array([all_states[k] for k in keys])
        
        from sklearn.decomposition import PCA
        if vecs.shape[0] >= 3:
            pca = PCA(n_components=2)
            vecs_2d = pca.fit_transform(vecs)
            
            ax2.scatter(vecs_2d[:, 0], vecs_2d[:, 1], c='#dfe6e9', s=20, alpha=0.5, zorder=1)
            
            # Current state
            cur_key = (pos[0], pos[1], facing)
            if cur_key in all_states:
                cur_idx = keys.index(cur_key)
                ax2.scatter(vecs_2d[cur_idx, 0], vecs_2d[cur_idx, 1], 
                           c=agent_color, s=80, zorder=3, edgecolors='white', linewidth=1.5)
            
            # Label some positions
            for i, k in enumerate(keys):
                y, x, f = k
                if vecs_2d[i, 0] > vecs_2d[:, 0].mean() + 0.5 or vecs_2d[i, 1] > vecs_2d[:, 1].mean() + 0.5:
                    ax2.annotate(f'({y},{x})', (vecs_2d[i, 0], vecs_2d[i, 1]),
                                fontsize=5, alpha=0.5, ha='center')
            
            ev = pca.explained_variance_ratio_
            ax2.set_title(f'State space (PCA: {ev[0]:.0%}+{ev[1]:.0%}={ev[:2].sum():.0%})', fontsize=10)
            ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
        else:
            ax2.text(0.5, 0.5, 'Need more positions visited', transform=ax2.transAxes, ha='center')
    else:
        ax2.text(0.5, 0.5, 'Collecting states...', transform=ax2.transAxes, ha='center')
    
    plt.tight_layout()
    plt.pause(0.05)
    
    if maze[pos] == CELL_EXIT:
        ax1.set_title(f'🎉 Reached exit! {step+1} steps', fontsize=14, color=exit_color, fontweight='bold')
        fig.canvas.draw()
        plt.pause(2)
        break

plt.ioff()
print(f'Done. {step+1} steps, visited {len(all_states)} state-position pairs.')
plt.show()
