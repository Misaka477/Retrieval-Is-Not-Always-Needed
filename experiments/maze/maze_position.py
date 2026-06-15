"""GRU maze model: predict next observation + position delta (Δy, Δx).

Position delta requires the model to understand spatial movement,
which CANNOT be solved by local pattern matching.
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

def av_acts(maze, pos, facing):
    acts=[0,1]
    dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL: acts.append(2)
    return acts

class PosGRU(nn.Module):
    """GRU that predicts next obs + position delta."""
    def __init__(self):
        super().__init__()
        self.obs_encoder = nn.Linear(12, DM)
        self.act_embed = nn.Embedding(3, DM//2)
        self.cell = nn.GRUCell(DM + DM//2, DM)
        self.obs_predictor = nn.Linear(DM, 12)
        self.pos_predictor = nn.Linear(DM, 2)  # predict (Δy, Δx)
        
    def forward(self, obs, act, h=None):
        oe = self.obs_encoder(obs); ae = self.act_embed(act)
        inp = torch.cat([oe, ae.squeeze(1)], -1)
        if h is None: h = torch.zeros(obs.size(0), DM, device=DEVICE)
        h = self.cell(inp, h)
        pred_obs = self.obs_predictor(h)
        pred_delta = self.pos_predictor(h)
        return pred_obs, pred_delta, h

model = PosGRU().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ── Data with position deltas ──
ALL_MAZES = [generate_maze() for _ in range(20)]

def gen_data(n_trajs=4000, traj_len=20):
    all_obs, all_act, all_next, all_delta, all_pos = [], [], [], [], []
    for _ in range(n_trajs):
        maze = random.choice(ALL_MAZES)
        h,w = maze.shape
        pos = (random.randrange(0,h), random.randrange(0,w))
        while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
        facing = random.randint(0,3)
        for _ in range(traj_len):
            o = get_obs(maze,pos,facing)
            all_obs.append(o); all_pos.append(pos)
            acts = av_acts(maze,pos,facing)
            act = random.choice(acts)
            old_pos = pos
            if act==2:
                dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
                if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL: pos=(ny,nx)
            elif act==0: facing=(facing-1)%4
            else: facing=(facing+1)%4
            delta = (pos[0]-old_pos[0], pos[1]-old_pos[1])
            all_act.append(act)
            all_next.append(get_obs(maze,pos,facing))
            all_delta.append(delta)
    return (torch.tensor(np.array(all_obs),device=DEVICE).float(),
            torch.tensor(all_act,device=DEVICE).long(),
            torch.tensor(np.array(all_next),device=DEVICE).float(),
            torch.tensor(all_delta,device=DEVICE).float())

OBS, ACT, NXT, DELTA = gen_data()
N_SAMPLES = OBS.size(0)
print(f'Data: {N_SAMPLES} transitions')
print(f'Delta range: y=[{DELTA[:,0].min().item():.0f},{DELTA[:,0].max().item():.0f}] '
      f'x=[{DELTA[:,1].min().item():.0f},{DELTA[:,1].max().item():.0f}]')

# ════════════════════════════════════════
# Train
# ════════════════════════════════════════

BSZ, N_STEPS = 256, 8000
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
TEST_MAZE = generate_maze(13, 13)

POS_WEIGHT = 0.5  # weight for position loss

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    idx = torch.randint(0, N_SAMPLES-BSZ, (BSZ,))
    obs = OBS[idx]; act = ACT[idx]; nxt = NXT[idx]; delta = DELTA[idx]
    
    pred_obs, pred_delta, _ = model(obs, act)
    obs_loss = F.mse_loss(pred_obs, nxt)
    pos_loss = F.mse_loss(pred_delta, delta)
    loss = obs_loss + POS_WEIGHT * pos_loss
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            idx_t = torch.randint(0, N_SAMPLES-256, (256,))
            po, pd, _ = model(OBS[idx_t], ACT[idx_t])
            ol = F.mse_loss(po, NXT[idx_t]).item()
            pl = F.mse_loss(pd, DELTA[idx_t]).item()
            acc = (po.round().clamp(0,1) == NXT[idx_t]).float().mean().item()
            
            # Full maze state encoding
            states = {}
            for y in range(TEST_MAZE.shape[0]):
                for x in range(TEST_MAZE.shape[1]):
                    if TEST_MAZE[y,x]==CELL_WALL: continue
                    for f in range(4):
                        o = get_obs(TEST_MAZE,(y,x),f)
                        ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                        _, _, h = model(ot, torch.tensor([[2]], device=DEVICE))
                        states[(y,x,f)] = h.squeeze(0).cpu().numpy()
            
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
        pbar.set_postfix(ol=f'{ol:.4f}', pl=f'{pl:.4f}',
                         acc=f'{acc*100:.0f}%', ratio=f'{ratio:.2f}x')

print(f'Train: {(time.time()-t0)/60:.1f}min')

# ════════════════════════════════════════
# Final eval
# ════════════════════════════════════════

print('\n=== Final ===')
model.eval()
with torch.no_grad():
    states = {}
    for y in range(TEST_MAZE.shape[0]):
        for x in range(TEST_MAZE.shape[1]):
            if TEST_MAZE[y,x]==CELL_WALL: continue
            for f in range(4):
                o = get_obs(TEST_MAZE,(y,x),f)
                ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                _, _, h = model(ot, torch.tensor([[2]], device=DEVICE))
                states[(y,x,f)] = h.squeeze(0).cpu().numpy()
    
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
    
    print(f'  State ratio: {np.mean(rand_d)/max(np.mean(adj_d),1e-8):.2f}x')
    print(f'  (GRU without pos: 1.00x | GRU with pos: ?? )')
    
    # PCA
    pca = PCA(n_components=2)
    s2d = pca.fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(8,6))
    colors = [(k[0]/13, k[1]/13, 0.5) for k in keys]
    ax.scatter(s2d[:,0], s2d[:,1], c=colors, s=15, alpha=0.7)
    ax.set_title(f'GRU+pos (PCA {pca.explained_variance_ratio_.sum():.0%})')
    plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_position.png'), dpi=150)
    print('  Saved maze_position.png')

    # Position delta prediction accuracy
    idx_test = torch.randint(0, N_SAMPLES-256, (256,))
    _, pd, _ = model(OBS[idx_test], ACT[idx_test])
    pd_acc = ((pd - DELTA[idx_test]).abs() < 0.5).float().mean().item()
    print(f'  Pos delta acc (within 0.5): {pd_acc*100:.1f}%')

torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_position.pt'))
print('Done.')
