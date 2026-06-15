"""Generate animated GIF of maze navigation with model predictions.
"""
import os, sys, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DM = 64
CELL_WALL, CELL_PATH, CELL_EXIT = 0, 1, 2
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Maze generator ──
def generate_maze(h=11, w=11):
    maze = np.ones((h, w), dtype=np.int32) * CELL_WALL
    start = (1, 1); maze[start] = CELL_PATH
    stack = [start]; rng = random.Random(42)
    while stack:
        cy, cx = stack[-1]
        nbs = []
        for dy,dx in [(-2,0),(2,0),(0,-2),(0,2)]:
            ny, nx = cy+dy, cx+dx
            if 0<ny<h-1 and 0<nx<w-1 and maze[ny,nx]==CELL_WALL:
                nbs.append((ny,nx))
        if nbs:
            ny, nx = rng.choice(nbs)
            maze[cy+(ny-cy)//2][cx+(nx-cx)//2] = CELL_PATH
            maze[ny,nx] = CELL_PATH; stack.append((ny,nx))
        else: stack.pop()
    maze[h-2,w-2] = CELL_EXIT
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        if maze[h-2+dy,w-2+dx] == CELL_WALL:
            maze[h-2+dy,w-2+dx] = CELL_PATH
    return maze

def get_obs(maze, pos, facing):
    obs = []
    for i in range(4):
        dy,dx = DIRS[(facing+i)%4]; ny,nx = pos[0]+dy, pos[1]+dx
        cell = maze[ny,nx] if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] else CELL_WALL
        o = [0,0,0]; o[cell] = 1; obs.extend(o)
    return np.array(obs, dtype=np.float32)

def av_acts(maze, pos, facing):
    acts = [0, 1]
    dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL:
        acts.append(2)
    return acts

# ── Model ──
class MazeWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder = nn.Linear(12, DM)
        self.act_embed = nn.Embedding(3, DM//2)
        self.cell = nn.GRUCell(DM + DM//2, DM)
        self.obs_predictor = nn.Linear(DM, 12)
    def forward(self, obs, act, h=None):
        oe = self.obs_encoder(obs); ae = self.act_embed(act)
        inp = torch.cat([oe, ae.squeeze(1)], -1)
        if h is None: h = torch.zeros(obs.size(0), DM, device=DEVICE)
        h = self.cell(inp, h)
        return self.obs_predictor(h), h

model = MazeWM().to(DEVICE)
ckpt = torch.load(os.path.join(BASE_DIR, 'checkpoints', 'maze_wm.pt'), weights_only=False, map_location=DEVICE)
model.load_state_dict(ckpt['model']); model.eval()

# ── Navigation ──
maze = generate_maze(11, 11)
h, w = maze.shape
pos, facing = (1, 1), random.randint(0, 3)
path = [pos]

frames = []
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

cmap = ListedColormap(['#2d2d2d', '#f0f0f0', '#4ecdc4'])
norm = plt.Normalize(0, 2)

for step in range(50):
    ax.clear()
    # Draw maze
    ax.imshow(maze, cmap=cmap, norm=norm, interpolation='nearest', alpha=0.9)
    
    # Draw path
    if len(path) > 1:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 1], path_arr[:, 0], 'b-', alpha=0.5, linewidth=2)
    
    # Agent
    ax.plot(pos[1], pos[0], 'o', markersize=14, color='#ff6b6b', zorder=5)
    # Direction indicator
    dy, dx = DIRS[facing]
    ax.arrow(pos[1], pos[0], dx*0.3, dy*0.3, head_width=0.3, 
             head_length=0.3, fc='#ff6b6b', ec='#ff6b6b', zorder=6)
    
    # Exit
    exit_y, exit_x = np.where(maze == CELL_EXIT)
    if len(exit_y):
        ax.plot(exit_x[0], exit_y[0], '*', markersize=18, color='#ffd93d', zorder=4)
    
    # Info
    ax.set_title(f'Step {step+1}/50  |  Pos: {pos}  Face: {["N","E","S","W"][facing]}', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    
    # Decide action: try to move toward exit
    exit_y, exit_x = np.where(maze == CELL_EXIT)
    target_y, target_x = (exit_y[0], exit_x[0]) if len(exit_y) else (h//2, w//2)
    
    acts = av_acts(maze, pos, facing)
    if 2 in acts:  # forward is valid
        dy, dx = DIRS[facing]
        ny, nx = pos[0]+dy, pos[1]+dx
        # Move toward exit if possible
        dist_now = abs(pos[0]-target_y) + abs(pos[1]-target_x)
        dist_next = abs(ny-target_y) + abs(nx-target_x)
        act = 2 if dist_next <= dist_now else random.choice(acts)
    else:
        # Rotate toward exit
        act = 0  # turn left
    
    # Use model to predict
    obs = get_obs(maze, pos, facing)
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred, h_state = model(obs_t, torch.tensor([[act]], device=DEVICE))
    
    # Show prediction
    pred_cells = pred.squeeze(0).cpu().numpy().reshape(4, 3).argmax(axis=1)
    cell_labels = ['WALL', 'PATH', 'EXIT']
    dir_labels = ['F', 'R', 'B', 'L']
    pred_text = '  '.join(f'{d}={cell_labels[p]}' for d, p in zip(dir_labels, pred_cells))
    ax.text(0.5, -0.05, f'Pred: {pred_text}', transform=ax.transAxes,
            fontsize=8, ha='center', va='top', family='monospace')
    
    frame_text = f'h_norm={h_state.norm().item():.2f}'
    ax.text(0.5, -0.1, frame_text, transform=ax.transAxes,
            fontsize=8, ha='center', va='top')
    
    # Execute
    dy, dx = DIRS[facing]
    if act == 2:
        ny, nx = pos[0]+dy, pos[1]+dx
        if maze[ny,nx] != CELL_WALL:
            pos = (ny, nx)
    elif act == 0: facing = (facing - 1) % 4
    else: facing = (facing + 1) % 4
    
    path.append(pos)
    
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frames.append(frame)
    
    if maze[pos] == CELL_EXIT:
        # Draw exit celebration
        ax.clear()
        ax.imshow(maze, cmap=cmap, norm=norm, interpolation='nearest', alpha=0.9)
        path_arr = np.array(path)
        ax.plot(path_arr[:, 1], path_arr[:, 0], 'b-', alpha=0.5, linewidth=2)
        ax.plot(pos[1], pos[0], 'o', markersize=20, color='#ffd93d', zorder=5)
        exit_y, exit_x = np.where(maze == CELL_EXIT)
        if len(exit_y): ax.plot(exit_x[0], exit_y[0], '*', markersize=18, color='#ffd93d', zorder=4)
        ax.set_title(f'🎉 Reached exit in {step+1} steps!', fontsize=14, color='#4ecdc4', fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        for _ in range(10): frames.append(frame)
        break

plt.close()

# ── Make GIF ──
from PIL import Image
gif_path = os.path.join(BASE_DIR, 'experiments', 'maze_nav.gif')
pil_frames = [Image.fromarray(f[..., :3]) for f in frames]
pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], 
                   duration=150, loop=0, optimize=False)
print(f'Saved {len(frames)} frames to {gif_path}')
