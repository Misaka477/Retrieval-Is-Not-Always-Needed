"""Watch the Transformer navigate a maze, step by step.
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

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_embed = nn.Linear(12, DM)
        self.pos_enc = PositionalEncoding(DM)
        encoder_layer = nn.TransformerEncoderLayer(d_model=DM, nhead=4, dim_feedforward=DM*2,
                                                    dropout=0.0, activation='gelu',
                                                    batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.predictor = nn.Linear(DM, 12)
        self.norm = nn.LayerNorm(DM)
    def forward(self, obs_seq):
        x = self.obs_embed(obs_seq)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.predictor(x)

model = MiniTransformer().to(DEVICE)
ckpt = torch.load(os.path.join(BASE_DIR, 'checkpoints', 'maze_transformer.pt'), weights_only=False, map_location=DEVICE)
model.load_state_dict(ckpt['model']); model.eval()
print('Loaded Transformer checkpoint')

# ── Live navigation ──
maze = generate_maze(13, 13)
h, w = maze.shape
pos, facing = (1, 1), 0
path = [pos]
obs_history = [get_obs(maze, pos, facing)]

plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
cmap = ListedColormap(['#2d2d2d', '#f0f0f0', '#4ecdc4'])

for step in range(100):
    acts = av_acts(maze, pos, facing)
    
    # Transformer predicts
    seq_t = torch.from_numpy(np.array(obs_history)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_all = model(seq_t)
        pred_next = pred_all[0, -1].cpu().numpy()
    
    # Pick action: compare predicted next obs for each action
    best_act, best_score = acts[0], -999
    for a in acts:
        # What would we see after this action?
        obs = get_obs(maze, pos, facing)
        if a == 2:
            dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
            no = get_obs(maze, (ny,nx), facing) if 0<=ny<h and 0<=nx<w else obs
        elif a == 0: no = get_obs(maze, pos, (facing-1)%4)
        else: no = get_obs(maze, pos, (facing+1)%4)
        
        score = -np.linalg.norm(pred_next - no)
        if score > best_score: best_act, best_score = a, score
    
    # Execute best action
    if best_act == 2:
        dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
        if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL: pos = (ny,nx)
    elif best_act == 0: facing = (facing-1)%4
    else: facing = (facing+1)%4
    
    path.append(pos)
    obs_history.append(get_obs(maze, pos, facing))
    
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
    
    status = '✅ EXIT!' if maze[pos]==CELL_EXIT else f'Step {step+1}  |  Transformer'
    dist = abs(pos[0]-11) + abs(pos[1]-11)
    ax.set_title(f'{status}  |  Dist from exit: {dist}', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    
    # POV
    dir_l = ['↑','→','↓','←']
    obs = get_obs(maze, pos, facing)
    cell_c = {CELL_WALL: '██', CELL_PATH: '··', CELL_EXIT: 'EE'}
    pov = '  '.join(f'{l}:{cell_c[np.argmax(obs[i*3:(i+1)*3])]}' for i,l in enumerate(dir_l))
    pred_c = {0:'██',1:'··',2:'EE'}
    pred_text = '  '.join(f'{l}:{pred_c[np.argmax(pred_next[i*3:(i+1)*3])]}' for i,l in enumerate(dir_l))
    ax.text(0.5, -0.03, f'POV: {pov}', transform=ax.transAxes, fontsize=8, ha='center', family='monospace')
    ax.text(0.5, -0.07, f'Pred: {pred_text}', transform=ax.transAxes, fontsize=8, ha='center', family='monospace', color='#0984e3')
    
    plt.pause(0.15)
    if maze[pos] == CELL_EXIT: break

print(f'Done. {step+1} steps, Dist={dist}')
plt.ioff(); plt.show()
