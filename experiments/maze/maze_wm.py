"""Maze World Model: test if WKV learns spatial structure.

Agent navigates a maze, observes 4 surrounding cells.
Model predicts next observation given current obs + action.
If the state space learns maze topology, the model understands space.

Loss: MSE(next_obs_pred, actual_next_obs) — pure prediction error
Verification: do states of nearby positions cluster together?
"""
import os, sys, time, math, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DM = 64  # small dim for fast experiment
print(f'Device: {DEVICE}  DM={DM}')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════
# Maze generator
# ═══════════════════════════════════════════════════

CELL_WALL, CELL_PATH, CELL_EXIT = 0, 1, 2
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W

def generate_maze(h=9, w=9):
    """Generate maze using DFS (odd dimensions for proper walls)."""
    maze = np.ones((h, w), dtype=np.int32) * CELL_WALL
    start = (1, 1)
    maze[start] = CELL_PATH
    stack = [start]
    rng = random.Random(42)
    
    while stack:
        cy, cx = stack[-1]
        nbs = []
        for dy, dx in [(-2,0),(2,0),(0,-2),(0,2)]:
            ny, nx = cy + dy, cx + dx
            if 0 < ny < h-1 and 0 < nx < w-1 and maze[ny, nx] == CELL_WALL:
                nbs.append((ny, nx))
        if nbs:
            ny, nx = rng.choice(nbs)
            maze[cy + (ny-cy)//2][cx + (nx-cx)//2] = CELL_PATH
            maze[ny, nx] = CELL_PATH
            stack.append((ny, nx))
        else:
            stack.pop()
    
    # Place exit at far corner
    exit_pos = (h-2, w-2)
    maze[exit_pos] = CELL_EXIT
    # Ensure reachable
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        if maze[exit_pos[0]+dy, exit_pos[1]+dx] == CELL_WALL:
            maze[exit_pos[0]+dy, exit_pos[1]+dx] = CELL_PATH
    
    return maze

def get_obs(maze, pos, facing):
    """Observation: 4 cells (front, right, back, left) encoded as 3-hot (wall/path/exit)."""
    obs = []
    for i in range(4):
        dir_idx = (facing + i) % 4
        dy, dx = DIRS[dir_idx]
        ny, nx = pos[0] + dy, pos[1] + dx
        if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1]:
            cell = maze[ny, nx]
        else:
            cell = CELL_WALL
        one_hot = [0, 0, 0]
        one_hot[cell] = 1
        obs.extend(one_hot)
    return np.array(obs, dtype=np.float32)  # 12-dim

def available_actions(maze, pos, facing):
    """Returns valid actions: 0=left, 1=right, 2=forward."""
    acts = [0, 1]  # rotate always possible
    dy, dx = DIRS[facing]
    ny, nx = pos[0] + dy, pos[1] + dx
    if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] != CELL_WALL:
        acts.append(2)
    return acts

def apply_action(maze, pos, facing, action):
    """Returns new_pos, new_facing, reward."""
    if action == 0:  # left
        return pos, (facing - 1) % 4, 0
    elif action == 1:  # right
        return pos, (facing + 1) % 4, 0
    else:  # forward
        dy, dx = DIRS[facing]
        ny, nx = pos[0] + dy, pos[1] + dx
        if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] != CELL_WALL:
            reward = 1.0 if maze[ny, nx] == CELL_EXIT else 0.0
            return (ny, nx), facing, reward
        return pos, facing, -0.1  # bumped into wall

# ═══════════════════════════════════════════════════
# Generate maze dataset
# ═══════════════════════════════════════════════════

print('Generating maze data...')
N_MAZES = 50
all_mazes = [generate_maze() for _ in range(N_MAZES)]

# Collect random trajectories
N_TRAJS = 5000
TRAJ_LEN = 32
all_obs, all_acts, all_next_obs, all_pos = [], [], [], []

for _ in range(N_TRAJS):
    maze = random.choice(all_mazes)
    h, w = maze.shape
    pos = (random.randrange(0, h), random.randrange(0, w))
    while maze[pos] == CELL_WALL:
        pos = (random.randrange(0, h), random.randrange(0, w))
    facing = random.randint(0, 3)
    
    for _ in range(TRAJ_LEN):
        obs = get_obs(maze, pos, facing)
        acts = available_actions(maze, pos, facing)
        act = random.choice(acts)
        new_pos, new_facing, _ = apply_action(maze, pos, facing, act)
        next_obs = get_obs(maze, new_pos, new_facing)
        
        all_obs.append(obs)
        all_acts.append(act)
        all_next_obs.append(next_obs)
        all_pos.append(pos)
        
        pos, facing = new_pos, new_facing

obs_t = torch.tensor(all_obs, device=DEVICE)
act_t = torch.tensor(all_acts, device=DEVICE, dtype=torch.long)
next_obs_t = torch.tensor(all_next_obs, device=DEVICE)
pos_t = torch.tensor(all_pos, device=DEVICE)
N = obs_t.size(0)
print(f'Dataset: {N} transitions, obs_dim=12, actions=3')

# ═══════════════════════════════════════════════════
# World Model
# ═══════════════════════════════════════════════════

class MazeWM(nn.Module):
    """Predict next observation given current obs + action."""
    def __init__(self):
        super().__init__()
        self.obs_encoder = nn.Linear(12, DM)
        self.act_embed = nn.Embedding(3, DM // 2)
        self.cell = nn.GRUCell(DM + DM // 2, DM)
        self.obs_predictor = nn.Linear(DM, 12)
        
    def forward(self, obs, act, h=None):
        obs_enc = self.obs_encoder(obs)
        act_enc = self.act_embed(act)
        inp = torch.cat([obs_enc, act_enc], dim=-1)
        if h is None:
            h = torch.zeros(obs.size(0), DM, device=DEVICE)
        h = self.cell(inp, h)
        pred = self.obs_predictor(h)
        return pred, h
    
    @torch.no_grad()
    def encode_world(self, maze):
        """Encode full maze by traversing all positions."""
        h_map = {}
        for y in range(maze.shape[0]):
            for x in range(maze.shape[1]):
                if maze[y, x] == CELL_WALL: continue
                for f in range(4):
                    obs = get_obs(maze, (y, x), f)
                    obs_t_ = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
                    # Just encode, no action needed
                    obs_enc = self.obs_encoder(obs_t_)
                    zero_act = torch.zeros(1, DM // 2, device=DEVICE)
                    inp = torch.cat([obs_enc, zero_act], dim=-1)
                    h = self.cell(inp, torch.zeros(1, DM, device=DEVICE))
                    h_map[(y, x, f)] = h.squeeze(0).cpu().numpy()
        return h_map

model = MazeWM().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ═══════════════════════════════════════════════════
# Train: predict next observation
# ═══════════════════════════════════════════════════

BSZ, N_STEPS, LR = 256, 10000, 3e-3
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()
history = []

for step in pbar:
    idx = torch.randint(0, N - BSZ, (BSZ,))
    obs = obs_t[idx]
    act = act_t[idx]
    target = next_obs_t[idx]
    
    pred, _ = model(obs, act)
    loss = F.mse_loss(pred, target)
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            idx_e = torch.randint(0, N - 256, (256,))
            po, pa, pt = obs_t[idx_e], act_t[idx_e], next_obs_t[idx_e]
            pp, _ = model(po, pa)
            mse = F.mse_loss(pp, pt).item()
            acc = (pp.round().clamp(0, 1) == pt).float().mean().item()
        model.train()
        history.append((step, mse, acc))
        pbar.set_postfix(mse=f'{loss.item():.6f}', tmse=f'{mse:.6f}', acc=f'{acc:.3f}',
                         lr=f'{sched.get_last_lr()[0]:.2e}')

print(f'Train: {(time.time()-t0)/60:.1f}min, final loss={loss.item():.6f}')

# ═══════════════════════════════════════════════════
# Verification: state space vs maze topology
# ═══════════════════════════════════════════════════

print('\n=== Verification ===')
model.eval()
with torch.no_grad():
    # Check prediction accuracy on held-out maze
    test_maze = generate_maze(11, 11)
    all_t_obs, all_t_next = [], []
    for y in range(test_maze.shape[0]):
        for x in range(test_maze.shape[1]):
            if test_maze[y, x] == CELL_WALL: continue
            for f in range(4):
                o = get_obs(test_maze, (y, x), f)
                dy, dx = DIRS[f]
                ny, nx = y + dy, x + dx
                if 0 <= ny < test_maze.shape[0] and 0 <= nx < test_maze.shape[1] and test_maze[ny, nx] != CELL_WALL:
                    no = get_obs(test_maze, (ny, nx), f)
                    all_t_obs.append(o)
                    all_t_next.append(no)
    
    if all_t_obs:
        to = torch.tensor(all_t_obs, device=DEVICE)
        tn = torch.tensor(all_t_next, device=DEVICE)
        ta = torch.ones(to.size(0), dtype=torch.long, device=DEVICE) * 2
        tp, _ = model(to, ta)
        mse = F.mse_loss(tp, tn).item()
        acc = (tp.round().clamp(0,1) == tn).float().mean().item()
        print(f'  Held-out maze: MSE={mse:.6f}  Acc={acc*100:.1f}%')
    
    # Encode the test maze and visualize structure
    h_map = model.encode_world(test_maze)
    
    # Reduce to 2D for visualization
    states = np.array(list(h_map.values()))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot maze
    ax = axes[0]
    maze_disp = np.zeros_like(test_maze, dtype=float)
    maze_disp[test_maze == CELL_WALL] = 1
    maze_disp[test_maze == CELL_EXIT] = 0.5
    ax.imshow(maze_disp, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Maze (white=wall, gray=exit)')
    
    # Plot state clusters by grid position
    ax = axes[1]
    h_pos = {k: v for k, v in h_map.items()}
    h_list = list(h_pos.values())
    if h_list:
        h_arr = np.array(h_list)
        from sklearn.decomposition import PCA as PCA2
        pca2 = PCA2(n_components=2)
        h_2d = pca2.fit_transform(h_arr)
        
        keys = list(h_pos.keys())
        for i, (k, v) in enumerate(zip(keys, h_2d)):
            y, x, f = k
            ax.scatter(v[0], v[1], c=[[y/11, x/11, 0.5]], s=20, alpha=0.6)
            ax.text(v[0], v[1], f'({y},{x})', fontsize=6, ha='center')
        
        ax.set_title(f'State space (PCA, explained var={pca2.explained_variance_ratio_.sum():.2f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_states.png'), dpi=150)
    print(f'  Saved maze_states.png')
    
    # Key metric: are nearby positions close in state space?
    keys = list(h_pos.keys())
    if len(keys) > 2:
        distances = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                y1, x1, _ = k1
                y2, x2, _ = k2
                if abs(y1-y2) + abs(x1-x2) == 1:  # adjacent cells
                    d = np.linalg.norm(h_pos[k1] - h_pos[k2])
                    distances.append(d)
        if distances:
            print(f'  Adjacent state distance: mean={np.mean(distances):.3f} std={np.std(distances):.3f}')
            
            # Compare with random pairs
            rand_dists = []
            for _ in range(1000):
                ki, kj = random.sample(keys, 2)
                rand_dists.append(np.linalg.norm(h_pos[ki] - h_pos[kj]))
            print(f'  Random pair distance: mean={np.mean(rand_dists):.3f} std={np.std(rand_dists):.3f}')
            print(f'  Separation (random/adjacent): {np.mean(rand_dists)/max(np.mean(distances),1e-8):.2f}x')

torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_wm.pt'))
print('\nDone.')
