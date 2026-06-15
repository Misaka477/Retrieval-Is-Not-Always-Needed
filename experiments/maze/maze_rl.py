"""GRU maze model: predict next obs + position delta + state VALUE.

V(s) = -distance_to_exit  (or +1000 at exit)
During navigation: pick action whose next state has highest V(s).

This way the model learns NOT JUST to predict, but to VALUE states
by how close they are to the exit.
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

class ValueGRU(nn.Module):
    """GRU that predicts next obs + position delta + state value V(s)."""
    def __init__(self):
        super().__init__()
        self.obs_encoder = nn.Linear(12, DM)
        self.act_embed = nn.Embedding(3, DM//2)
        self.cell = nn.GRUCell(DM + DM//2, DM)
        self.obs_predictor = nn.Linear(DM, 12)
        self.pos_predictor = nn.Linear(DM, 2)
        self.value_head = nn.Linear(DM, 1)  # V(s) = how good is this state
        
    def forward(self, obs, act, h=None):
        oe = self.obs_encoder(obs); ae = self.act_embed(act)
        inp = torch.cat([oe, ae.squeeze(1)], -1)
        if h is None: h = torch.zeros(obs.size(0), DM, device=DEVICE)
        h = self.cell(inp, h)
        pred_obs = self.obs_predictor(h)
        pred_delta = self.pos_predictor(h)
        value = self.value_head(h).squeeze(-1)  # [B]
        return pred_obs, pred_delta, value, h

model = ValueGRU().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ── Data with position deltas + exit distances ──
ALL_MAZES = [generate_maze() for _ in range(20)]

def gen_data(n_trajs=4000, traj_len=20):
    all_obs, all_act, all_next, all_delta, all_value = [], [], [], [], []
    all_pos = []
    for _ in range(n_trajs):
        maze = random.choice(ALL_MAZES)
        h,w = maze.shape
        exit_pos = (h-2, w-2)  # bottom-right
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
            
            # Value = -distance_to_exit, or +1000 for exit
            dist = abs(pos[0]-exit_pos[0]) + abs(pos[1]-exit_pos[1])
            val = 1000.0 if maze[pos]==CELL_EXIT else -float(dist)
            val = val / 10.0  # normalize to roughly [-1.6, 100]
            
            delta = (pos[0]-old_pos[0], pos[1]-old_pos[1])
            all_act.append(act)
            all_next.append(get_obs(maze,pos,facing))
            all_delta.append(delta)
            all_value.append(val)
    return (torch.tensor(np.array(all_obs),device=DEVICE).float(),
            torch.tensor(all_act,device=DEVICE).long(),
            torch.tensor(np.array(all_next),device=DEVICE).float(),
            torch.tensor(all_delta,device=DEVICE).float(),
            torch.tensor(all_value,device=DEVICE).float())

OBS, ACT, NXT, DELTA, VAL = gen_data()
N_SAMPLES = OBS.size(0)
print(f'Data: {N_SAMPLES} transitions')
print(f'Value range: [{VAL.min().item():.0f}, {VAL.max().item():.0f}]')

# ════════════════════════════════════════
# Train
# ════════════════════════════════════════

BSZ, N_STEPS = 256, 8000
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
TEST_MAZE = generate_maze(13, 13)

POS_W, VAL_W = 0.5, 0.5  # value weight = observation weight

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    idx = torch.randint(0, N_SAMPLES-BSZ, (BSZ,))
    obs = OBS[idx]; act = ACT[idx]; nxt = NXT[idx]
    delta = DELTA[idx]; val = VAL[idx]
    
    pred_obs, pred_delta, pred_val, _ = model(obs, act)
    ol = F.mse_loss(pred_obs, nxt)
    pl = F.mse_loss(pred_delta, delta)
    vl = F.mse_loss(pred_val, val)
    loss = ol + POS_W * pl + VAL_W * vl
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            idx_t = torch.randint(0, N_SAMPLES-256, (256,))
            po, pd, pv, _ = model(OBS[idx_t], ACT[idx_t])
            tol = F.mse_loss(po, NXT[idx_t]).item()
            tpl = F.mse_loss(pd, DELTA[idx_t]).item()
            tvl = F.mse_loss(pv, VAL[idx_t]).item()
            tacc = (po.round().clamp(0,1) == NXT[idx_t]).float().mean().item()
            
            # State structure
            states = {}
            for y in range(TEST_MAZE.shape[0]):
                for x in range(TEST_MAZE.shape[1]):
                    if TEST_MAZE[y,x]==CELL_WALL: continue
                    for f in range(4):
                        o = get_obs(TEST_MAZE,(y,x),f)
                        ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                        _, _, _, h = model(ot, torch.tensor([[2]], device=DEVICE))
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
            
            # Value prediction accuracy: does V(s) correlate with distance to exit?
            exit_dist = [abs(k[0]-11)+abs(k[1]-11) for k in keys]
            vecs_t = torch.from_numpy(vecs).to(DEVICE)
            pred_vals = model.value_head(vecs_t).squeeze(-1).cpu().numpy()
            val_corr = np.corrcoef(pred_vals, exit_dist)[0,1] if len(exit_dist)>1 else 0.0
        
        model.train()
        pbar.set_postfix(ol=f'{tol:.4f}', pl=f'{tpl:.4f}', vl=f'{tvl:.4f}',
                         acc=f'{tacc*100:.0f}%', ratio=f'{ratio:.2f}x',
                         vcorr=f'{val_corr:.2f}')

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
                _, _, _, h = model(ot, torch.tensor([[2]], device=DEVICE))
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
    print(f'  State ratio: {ratio:.2f}x')
    
    # Value vs distance
    exit_dist = [abs(k[0]-11)+abs(k[1]-11) for k in keys]
    vecs_t = torch.from_numpy(vecs).to(DEVICE)
    pred_vals = model.value_head(vecs_t).squeeze(-1).cpu().numpy()
    val_corr = np.corrcoef(pred_vals, exit_dist)[0,1] if len(exit_dist)>1 else 0.0
    print(f'  Value ↔ egress distance corr: {val_corr:.2f}')
    
    # PCA
    pca = PCA(n_components=2)
    s2d = pca.fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(s2d[:,0], s2d[:,1], c=pred_vals, cmap='RdYlGn', s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='V(s)')
    ax.set_title(f'Value-Conditioned States (PCA {pca.explained_variance_ratio_.sum():.0%})')
    plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_rl.png'), dpi=150)
    print('  Saved maze_rl.png')

torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_rl.pt'))
print('Done.')
