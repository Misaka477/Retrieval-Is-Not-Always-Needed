"""Maze world model + distance-aware contrastive loss.

Train: predict next observation (MSE) + force state space to encode maze topology.
Contrastive: cos(h_i, h_j) ≈ exp(-dist(pos_i, pos_j) / τ)
   → nearby positions have similar states
   → far positions have different states

If this works: state distance ratio (adjacent/random) >> 1.0, agent learns to navigate.
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

print(f'Device: {DEVICE}  DM={DM}')

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
    
    @torch.no_grad()
    def encode_maze(self, maze):
        states = {}
        for y in range(maze.shape[0]):
            for x in range(maze.shape[1]):
                if maze[y,x]==CELL_WALL: continue
                for f in range(4):
                    o=get_obs(maze,(y,x),f)
                    ot=torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                    _,h=self(ot,torch.tensor([[2]],device=DEVICE))
                    states[(y,x,f)]=h.squeeze(0).cpu().numpy()
        return states

model = MazeWM().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ── Data with positions ──
ALL_MAZES = [generate_maze() for _ in range(20)]

def gen_data(n_trajs=3000, traj_len=24):
    all_obs, all_act, all_next, all_pos = [], [], [], []
    for _ in range(n_trajs):
        maze = random.choice(ALL_MAZES)
        h,w = maze.shape
        pos = (random.randrange(0,h), random.randrange(0,w))
        while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
        facing = random.randint(0,3)
        for _ in range(traj_len):
            o=get_obs(maze,pos,facing)
            all_pos.append(pos)
            acts=av_acts(maze,pos,facing)
            act=random.choice(acts)
            if act==2:
                dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
                if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL: pos=(ny,nx)
            elif act==0: facing=(facing-1)%4
            else: facing=(facing+1)%4
            all_obs.append(o); all_act.append(act); all_next.append(get_obs(maze,pos,facing))
    return (torch.tensor(np.array(all_obs),device=DEVICE).float(),
            torch.tensor(all_act,device=DEVICE).long(),
            torch.tensor(np.array(all_next),device=DEVICE).float(),
            torch.tensor(all_pos,device=DEVICE).float())

OBS, ACT, NXT, POS = gen_data(3000, 24)
N_SAMPLES = OBS.size(0)
print(f'Data: {N_SAMPLES} transitions, pos_range=[{POS.min().item():.0f},{POS.max().item():.0f}]')

# ════════════════════════════════════════
# Distance-aware contrastive loss
# ════════════════════════════════════════

def contrastive_dist_loss(h, pos, tau=3.0, lambda_c=0.5):
    """Force state similarity to reflect spatial distance.
    
    Target: sim(h_i, h_j) = exp(-dist(pos_i, pos_j) / tau)
    Loss: MSE(actual_sim, target_sim)
    """
    N = h.size(0)
    
    # Compute pairwise distances
    pos_flat = pos  # [N, 2]
    d = torch.cdist(pos_flat, pos_flat, p=1)  # Manhattan distance [N, N]
    target_sim = torch.exp(-d / tau)  # [N, N]
    
    # Compute pairwise cosine similarity
    h_norm = F.normalize(h, dim=-1)  # [N, DM]
    actual_sim = h_norm @ h_norm.T  # [N, N]
    
    # MSE loss (mask diagonal)
    mask = ~torch.eye(N, dtype=torch.bool, device=DEVICE)
    loss = F.mse_loss(actual_sim[mask], target_sim[mask])
    
    with torch.no_grad():
        # Correlation between actual sim and target sim
        corr = torch.corrcoef(torch.stack([actual_sim[mask], target_sim[mask]]))[0, 1].item()
    
    return loss * lambda_c, corr

# ════════════════════════════════════════
# Train
# ════════════════════════════════════════

BSZ, N_STEPS = 256, 10000
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

# Test data
TEST_MAZE = generate_maze(13, 13)

for step in pbar:
    idx = torch.randint(0, N_SAMPLES-BSZ, (BSZ,))
    obs = OBS[idx]; act = ACT[idx]; target = NXT[idx]; pos = POS[idx]
    
    pred, h = model(obs, act)
    mse = F.mse_loss(pred, target)
    c_loss, corr = contrastive_dist_loss(h, pos)
    loss = mse + c_loss
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            idx_t = torch.randint(0, N_SAMPLES-256, (256,))
            to=OBS[idx_t]; ta=ACT[idx_t]; tn=NXT[idx_t]; tp=POS[idx_t]
            pp, ph = model(to, ta)
            t_mse = F.mse_loss(pp, tn).item()
            t_acc = (pp.round().clamp(0,1) == tn).float().mean().item()
            
            # State structure analysis
            states = model.encode_maze(TEST_MAZE)
            if states:
                keys = list(states.keys())
                vecs = np.array([states[k] for k in keys])
                # Adjacent vs random
                adj_dists, rand_dists = [], []
                for i in range(len(keys)):
                    for j in range(i+1, len(keys)):
                        k1,k2 = keys[i], keys[j]
                        md = abs(k1[0]-k2[0]) + abs(k1[1]-k2[1])
                        d = np.linalg.norm(vecs[i]-vecs[j])
                        if md == 1: adj_dists.append(d)
                        else: rand_dists.append(d)
                ratio = np.mean(rand_dists)/max(np.mean(adj_dists),1e-8) if adj_dists else 1.0
            else:
                ratio = 1.0
            
        model.train()
        pbar.set_postfix(mse=f'{mse.item():.4f}', cl=f'{c_loss.item():.4f}',
                         corr=f'{corr:.2f}', ratio=f'{ratio:.2f}x',
                         acc=f'{t_acc*100:.0f}%', lr=f'{sched.get_last_lr()[0]:.1e}')

elapsed = (time.time()-t0)/60
print(f'\nTrain: {elapsed:.1f}min, final loss={loss.item():.6f}')

# ════════════════════════════════════════
# Final evaluation
# ════════════════════════════════════════

print('\n=== Final Evaluation ===')
model.eval()
with torch.no_grad():
    idx_t = torch.randint(0, N_SAMPLES-256, (256,))
    to=OBS[idx_t]; ta=ACT[idx_t]; tn=NXT[idx_t]; tp=POS[idx_t]
    pp, ph = model(to, ta)
    mse = F.mse_loss(pp, tn).item()
    acc = (pp.round().clamp(0,1) == tn).float().mean().item()
    print(f'  Prediction: MSE={mse:.4f} Acc={acc*100:.1f}%')
    
    # State structure
    states = model.encode_maze(TEST_MAZE)
    keys = list(states.keys())
    vecs = np.array([states[k] for k in keys])
    adj_dists, rand_dists = [], []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1,k2 = keys[i], keys[j]
            md = abs(k1[0]-k2[0]) + abs(k1[1]-k2[1])
            d = np.linalg.norm(vecs[i]-vecs[j])
            if md == 1: adj_dists.append(d)
            elif md > 3: rand_dists.append(d)
    print(f'  Adjacent state distance: mean={np.mean(adj_dists):.3f}')
    print(f'  Random state distance: mean={np.mean(rand_dists):.3f}')
    print(f'  Ratio: {np.mean(rand_dists)/max(np.mean(adj_dists),1e-8):.2f}x')
    
    # PCA visualization
    pca = PCA(n_components=2)
    s2d = pca.fit_transform(vecs)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(TEST_MAZE, cmap=ListedColormap(['#2d2d2d','#f0f0f0','#4ecdc4']), vmin=0, vmax=2)
    ax[0].set_title('Test Maze', fontsize=11)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    
    positions = [(k[0], k[1]) for k in keys]
    colors = [(p[0]/TEST_MAZE.shape[0], p[1]/TEST_MAZE.shape[1], 0.5) for p in positions]
    ax[1].scatter(s2d[:,0], s2d[:,1], c=colors, s=20, alpha=0.7)
    ax[1].set_title(f'State Space (PCA {pca.explained_variance_ratio_.sum():.0%})', fontsize=11)
    ax[1].set_xlabel('PC1'); ax[1].set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_distloss.png'), dpi=150)
    print(f'  Saved maze_distloss.png')

torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_dist.pt'))
print('Done.')
