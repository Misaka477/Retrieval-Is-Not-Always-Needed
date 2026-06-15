"""Pure contrastive state space — no prediction, no MSE.

Model learns: state_distance ≈ spatial_distance.
Navigation: find exit by following the state-space gradient toward exit state.
"""
import os, sys, random, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
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
    return np.array(obs,dtype=np.float32)

# ── Pure state encoder (no prediction head) ──
class StateEncoder(nn.Module):
    """obs → state vector. That's it. No prediction."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, DM),
            nn.GELU(),
            nn.Linear(DM, DM),
            nn.GELU(),
            nn.Linear(DM, DM),
        )
        for m in self.net:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, 0.5)

    def forward(self, obs):
        return self.net(obs)  # [B, DM]

model = StateEncoder().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ════════════════════════════════════════
# Dataset: (obs, pos) pairs for every position in many mazes
# ════════════════════════════════════════

ALL_MAZES = [generate_maze() for _ in range(30)]
all_obs, all_pos = [], []

for maze in ALL_MAZES:
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y,x] != CELL_WALL:
                for f in range(4):
                    all_obs.append(get_obs(maze,(y,x),f))
                    all_pos.append((y,x))

OBS = torch.tensor(np.array(all_obs), device=DEVICE).float()
POS = torch.tensor(all_pos, device=DEVICE).float()
N = OBS.size(0)
print(f'Data: {N} (obs, pos) pairs')

# ════════════════════════════════════════
# Contrastive spatial loss
# ════════════════════════════════════════

def spatial_contrastive_loss(h, pos, tau=2.0):
    """State similarity must reflect spatial distance.
    loss = MSE(cos(h_i, h_j), exp(-dist(pos_i, pos_j) / tau))
    """
    N = h.size(0)
    d = torch.cdist(pos, pos, p=1)  # [N, N] Manhattan
    target = torch.exp(-d / tau)  # [N, N]
    
    h_n = F.normalize(h, dim=-1)
    actual = h_n @ h_n.T  # [N, N]
    
    mask = ~torch.eye(N, dtype=torch.bool, device=DEVICE)
    loss = F.mse_loss(actual[mask], target[mask])
    
    with torch.no_grad():
        corr = torch.corrcoef(
            torch.stack([actual[mask], target[mask]])
        )[0,1].item() if N > 3 else 0.0
    
    return loss, corr

# Also add uniformity loss: encourage state norms to spread out
def uniformity_loss(h):
    h_n = F.normalize(h, dim=-1)
    sim = h_n @ h_n.T
    # Push all pairs apart uniformly
    return sim[~torch.eye(h.size(0), dtype=torch.bool, device=DEVICE)].mean()

# ════════════════════════════════════════
# Train
# ════════════════════════════════════════

BSZ, N_STEPS = 512, 5000
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

TEST_MAZE = generate_maze(13, 13)

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    idx = torch.randint(0, N-BSZ, (BSZ,))
    obs = OBS[idx]; pos = POS[idx]
    
    h = model(obs)
    c_loss, corr = spatial_contrastive_loss(h, pos)
    u_loss = uniformity_loss(h)
    loss = c_loss + 0.1 * u_loss
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            idx_t = torch.randint(0, N-256, (256,))
            h_t = model(OBS[idx_t])
            tl, t_corr = spatial_contrastive_loss(h_t, POS[idx_t])
            
            # Encode test maze and check structure
            states = {}
            for y in range(TEST_MAZE.shape[0]):
                for x in range(TEST_MAZE.shape[1]):
                    if TEST_MAZE[y,x]==CELL_WALL: continue
                    for f in range(4):
                        o = get_obs(TEST_MAZE,(y,x),f)
                        ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                        h_test = model(ot)
                        states[(y,x,f)] = h_test.squeeze(0).cpu().numpy()
            
            keys = list(states.keys())
            vecs = np.array([states[k] for k in keys])
            adj_d, rand_d = [], []
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    k1,k2 = keys[i], keys[j]
                    md = abs(k1[0]-k2[0]) + abs(k1[1]-k2[1])
                    d = np.linalg.norm(vecs[i]-vecs[j])
                    if md == 1: adj_d.append(d)
                    elif md > 4: rand_d.append(d)
            ratio = np.mean(rand_d)/max(np.mean(adj_d),1e-8) if adj_d and rand_d else 1.0
        
        model.train()
        pbar.set_postfix(cl=f'{c_loss.item():.4f}', corr=f'{corr:.2f}',
                         r=f'{ratio:.2f}x', lr=f'{sched.get_last_lr()[0]:.1e}')

print(f'Train: {(time.time()-t0)/60:.1f}min')

# ════════════════════════════════════════
# Final evaluation
# ════════════════════════════════════════

print('\n=== Final Evaluation ===')
model.eval()
with torch.no_grad():
    # Full state space structure
    states = {}
    for y in range(TEST_MAZE.shape[0]):
        for x in range(TEST_MAZE.shape[1]):
            if TEST_MAZE[y,x]==CELL_WALL: continue
            for f in range(4):
                o = get_obs(TEST_MAZE,(y,x),f)
                ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                states[(y,x,f)] = model(ot).squeeze(0).cpu().numpy()

    keys = list(states.keys())
    vecs = np.array([states[k] for k in keys])
    
    adj_d, rand_d = [], []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1,k2 = keys[i], keys[j]
            md = abs(k1[0]-k2[0]) + abs(k1[1]-k2[1])
            d = np.linalg.norm(vecs[i]-vecs[j])
            if md == 1: adj_d.append(d)
            elif md > 4: rand_d.append(d)
    
    print(f'  Adjacent distance: mean={np.mean(adj_d):.3f} std={np.std(adj_d):.3f}')
    print(f'  Random distance:  mean={np.mean(rand_d):.3f} std={np.std(rand_d):.3f}')
    print(f'  Ratio: {np.mean(rand_d)/max(np.mean(adj_d),1e-8):.2f}x')
    
    # PCA visualization
    pca = PCA(n_components=2)
    s2d = pca.fit_transform(vecs)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.imshow(TEST_MAZE, cmap=ListedColormap(['#2d2d2d','#f0f0f0','#4ecdc4']), vmin=0, vmax=2)
    ax.set_title('Test Maze', fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    
    ax = axes[1]
    colors = [(k[0]/13, k[1]/13, 0.5) for k in keys]
    sc = ax.scatter(s2d[:,0], s2d[:,1], c=colors, s=15, alpha=0.7)
    ax.set_title(f'State Space (PCA, ev={pca.explained_variance_ratio_.sum():.0%})', fontsize=10)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    
    ax = axes[2]
    # Distance correlation plot
    dists_maze = []
    dists_state = []
    for i in range(min(500, len(keys))):
        for j in range(i+1, min(500, len(keys))):
            k1,k2 = keys[i], keys[j]
            md = abs(k1[0]-k2[0]) + abs(k1[1]-k2[1])
            d = np.linalg.norm(vecs[i]-vecs[j])
            dists_maze.append(md)
            dists_state.append(d)
    ax.scatter(dists_maze, dists_state, s=1, alpha=0.3, c='#0984e3')
    ax.set_title('Spatial Dist vs State Dist', fontsize=10)
    ax.set_xlabel('Manhattan dist in maze'); ax.set_ylabel('Euclidean dist in state space')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_contrast.png'), dpi=150)
    print('  Saved maze_contrast.png')

torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_contrast.pt'))
print('Done.')
