"""Live training + live agent navigation: watch agent learn to walk the maze.

Every N steps: reset agent → use model predictions to navigate → show path.
As training progresses, the path should get better (more walls avoided).
"""
import os, sys, random, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DM = 64
CELL_WALL, CELL_PATH, CELL_EXIT = 0, 1, 2
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f'Device: {DEVICE}')

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

model = MazeWM().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

# ── Data ──
ALL_MAZES = [generate_maze() for _ in (range(20))]
# Fixed test maze for demo
TEST_MAZE = generate_maze(13, 13)
START = (1, 1)  # top-left path cell

def gen_data(n_trajs=3000, traj_len=24):
    all_obs, all_act, all_next = [], [], []
    for _ in range(n_trajs):
        maze = random.choice(ALL_MAZES)
        h,w = maze.shape
        pos = (random.randrange(0,h), random.randrange(0,w))
        while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
        facing = random.randint(0,3)
        for _ in range(traj_len):
            o=get_obs(maze,pos,facing)
            acts=av_acts(maze,pos,facing)
            act=random.choice(acts)
            if act==2:
                dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
                pos=(ny,nx)
            elif act==0: facing=(facing-1)%4
            else: facing=(facing+1)%4
            all_obs.append(o); all_act.append(act); all_next.append(get_obs(maze,pos,facing))
    return (torch.tensor(np.array(all_obs),device=DEVICE).float(),
            torch.tensor(all_act,device=DEVICE).long(),
            torch.tensor(np.array(all_next),device=DEVICE).float())

OBS, ACT, NXT = gen_data()
N_SAMPLES = OBS.size(0)
print(f'Data: {N_SAMPLES} transitions')

# ── Model-guided navigation ──
@torch.no_grad()
def navigate(model, maze, pos, facing, max_steps=100):
    """Use model to navigate:
    For each action: predict what we'd see → pick action with most PATH in front.
    """
    path = [pos]
    visits = {pos: 0}
    for s in range(max_steps):
        if maze[pos] == CELL_EXIT: break
        
        acts = av_acts(maze, pos, facing)
        best_act = acts[0]
        best_score = -999
        
        for a in acts:
            obs = get_obs(maze, pos, facing)
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
            at = torch.tensor([[a]], device=DEVICE)
            pred, _ = model(ot, at)
            pred_cells = pred.squeeze(0).cpu().numpy().reshape(4, 3)
            
            # Score: how much open space (PATH) would we see?
            open_score = pred_cells[0, CELL_PATH]  # front cell
            if a == 2:  # moving forward
                dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
                if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1]:
                    if maze[ny,nx] == CELL_EXIT: open_score = 10
            score = open_score - 0.1 * visits.get(tuple(pos), 0)
            
            if score > best_score:
                best_score = score
                best_act = a
        
        # Execute best action
        if best_act == 2:
            dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
            if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL:
                pos = (ny,nx)
        elif best_act == 0: facing = (facing-1)%4
        else: facing = (facing+1)%4
        
        path.append(pos)
        visits[pos] = visits.get(pos, 0) + 1
    
    return path, s+1

# ── Live plot ──
plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Training Maze World Model — Watch the Agent Learn', fontsize=14, fontweight='bold')
ax_loss = axes[0]
ax_snapshot = axes[1]
ax_stats = axes[2]
cmap_maze = ListedColormap(['#2d2d2d', '#f0f0f0', '#4ecdc4'])

loss_hist = []
pathlen_hist = []
step_hist = []

BSZ, N_STEPS = 256, 15000
UPDATE = 300

# Pre-compute a few colors for different training stages
stage_colors = ['#ff6b6b', '#fdcb6e', '#00b894', '#0984e3', '#6c5ce7', '#e17055', '#00cec9', '#a29bfe']

for step in range(N_STEPS):
    idx = torch.randint(0, N_SAMPLES-BSZ, (BSZ,))
    obs = OBS[idx]; act = ACT[idx]; target = NXT[idx]
    pred, _ = model(obs, act)
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    
    if step % UPDATE == 0:
        model.eval()
        with torch.no_grad():
            idx_t = torch.randint(0, N_SAMPLES-256, (256,))
            to = OBS[idx_t]; ta = ACT[idx_t]; tn = NXT[idx_t]
            tp, _ = model(to, ta)
            tloss = F.mse_loss(tp, tn).item()
            tacc = (tp.round().clamp(0,1) == tn).float().mean().item()
            
            # Agent navigates using current model
            path, n_steps = navigate(model, TEST_MAZE, START, 0)
            reached = TEST_MAZE[path[-1]] == CELL_EXIT
            exit_dist = abs(path[-1][0]-11) + abs(path[-1][1]-11)
            
            loss_hist.append(loss.item())
            pathlen_hist.append(n_steps)
            step_hist.append(step)
        
        # Loss
        ax_loss.clear()
        ax_loss.plot(step_hist, loss_hist, '-', color='#0984e3', linewidth=2)
        ax_loss.set_title('Loss (MSE) — lower = better', fontsize=10)
        ax_loss.set_xlabel('Step'); ax_loss.grid(True, alpha=0.2)
        ax_loss.text(0.95, 0.95, f'{loss_hist[-1]:.4f}', transform=ax_loss.transAxes,
                    va='top', ha='right', fontsize=12, fontweight='bold', color='#0984e3')
        
        # Path evolution: overlay paths from different stages
        ax_snapshot.clear()
        ax_snapshot.imshow(TEST_MAZE, cmap=cmap_maze, vmin=0, vmax=2, interpolation='nearest')
        
        # Show current path (thick, bright) and historical paths (thin, faded)
        n_hist = len(step_hist)
        c_idx = (n_hist - 1) % len(stage_colors)
        color = stage_colors[c_idx]
        
        pa = np.array(path)
        ax_snapshot.plot(pa[:,1], pa[:,0], '-', color=color, linewidth=2.5, 
                        alpha=0.8, zorder=4, label=f'Step {step}')
        ax_snapshot.plot(path[-1][1], path[-1][0], 'o', markersize=12, 
                        color=color, zorder=5)
        
        # Exit
        ey,ex = np.where(TEST_MAZE==CELL_EXIT)
        if len(ey): ax_snapshot.plot(ex[0], ey[0], '*', markersize=18, color='#ffd93d', zorder=3)
        
        # Start
        ax_snapshot.plot(START[1], START[0], 's', markersize=8, color='white', zorder=3)
        
        ax_snapshot.set_title(f'Agent navigation at step {step}\n'
                             f'{n_steps} steps | {"✅ Reached exit!" if reached else f"❌ {exit_dist} from exit"}',
                             fontsize=10)
        ax_snapshot.set_xticks([]); ax_snapshot.set_yticks([])
        ax_snapshot.legend(fontsize=7, loc='lower left')
        
        # Stats
        ax_stats.clear()
        ax_stats.axis('off')
        stats_text = (
            f'TRAINING PROGRESS\n'
            f'{"="*25}\n\n'
            f'Step:       {step}/{N_STEPS}\n'
            f'Loss:       {loss.item():.6f}\n'
            f'Test Loss:  {tloss:.6f}\n'
            f'Test Acc:   {tacc*100:.1f}%\n\n'
            f'AGENT NAVIGATION\n'
            f'{"="*25}\n\n'
            f'Steps taken: {n_steps}\n'
            f'Reached exit: {"✅ Yes" if reached else "❌ No"}\n'
            f'Distance to exit: {exit_dist}\n'
            f'Path length: {len(path)}\n'
            f'Start: {START}\n'
            f'End:   {path[-1]}\n\n'
            f'PATH HISTORY\n'
            f'{"="*25}\n'
        )
        for i, (s, pl) in enumerate(zip(step_hist[-10:], pathlen_hist[-10:])):
            color_star = '⭐' if i == len(pathlen_hist[-10:])-1 else '·'
            stats_text += f'\n{color_star} step {s}: {pl} steps'
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=9, va='top', ha='left', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))
        
        plt.tight_layout()
        plt.pause(0.05)
        model.train()

# Save model
ckpt_path = os.path.join(BASE_DIR, 'checkpoints', 'maze_trained.pt')
torch.save({'model': model.state_dict(), 'step': step, 'loss': loss.item()}, ckpt_path)
print(f'Saved to {ckpt_path}')
print(f'To watch agent navigate: python3 experiments/maze_viewer.py')

plt.ioff(); plt.show()
