"""GRU with (y, x) position in input. That's the only change.

Observation = 4 surrounding cells (12-dim) + position (2-dim, normalized) = 14-dim
If this works, the state space should encode spatial structure trivially.
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
OBS_DIM = 12 + 2  # 4 cells × 3 one-hot + (y, x) normalized
print(f'Device: {DEVICE}  OBS_DIM={OBS_DIM}')

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
    # Append normalized position
    h, w = maze.shape
    obs.append(pos[0] / h)  # y normalized
    obs.append(pos[1] / w)  # x normalized
    return np.array(obs, dtype=np.float32)

def av_acts(maze, pos, facing):
    acts=[0,1]
    dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL: acts.append(2)
    return acts

class PosInputGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder = nn.Linear(OBS_DIM, DM)
        self.act_embed = nn.Embedding(3, DM//2)
        self.cell = nn.GRUCell(DM + DM//2, DM)
        self.obs_predictor = nn.Linear(DM, OBS_DIM)
        
    def forward(self, obs, act, h=None):
        oe = self.obs_encoder(obs); ae = self.act_embed(act)
        inp = torch.cat([oe, ae.squeeze(1)], -1)
        if h is None: h = torch.zeros(obs.size(0), DM, device=DEVICE)
        h = self.cell(inp, h)
        return self.obs_predictor(h), h

model = PosInputGRU().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ── Data ──
ALL_MAZES = [generate_maze() for _ in range(15)]

def gen_data(n_trajs=3000, traj_len=20):
    all_obs, all_act, all_next = [], [], []
    for _ in range(n_trajs):
        maze = random.choice(ALL_MAZES)
        h,w = maze.shape
        pos = (random.randrange(0,h), random.randrange(0,w))
        while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
        facing = random.randint(0,3)
        for _ in range(traj_len):
            o = get_obs(maze,pos,facing)
            all_obs.append(o)
            acts = av_acts(maze,pos,facing)
            act = random.choice(acts)
            if act==2:
                dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
                if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL: pos=(ny,nx)
            elif act==0: facing=(facing-1)%4
            else: facing=(facing+1)%4
            all_act.append(act)
            all_next.append(get_obs(maze,pos,facing))
    return (torch.tensor(np.array(all_obs),device=DEVICE).float(),
            torch.tensor(all_act,device=DEVICE).long(),
            torch.tensor(np.array(all_next),device=DEVICE).float())

OBS, ACT, NXT = gen_data()
N_SAMPLES = OBS.size(0)
print(f'Data: {N_SAMPLES} transitions')

# ── Train ──
BSZ, N_STEPS = 256, 5000
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
TEST_MAZE = generate_maze(13, 13)

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    idx = torch.randint(0, N_SAMPLES-BSZ, (BSZ,))
    obs = OBS[idx]; act = ACT[idx]; target = NXT[idx]
    pred, _ = model(obs, act)
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            idx_t = torch.randint(0, N_SAMPLES-256, (256,))
            pp, _ = model(OBS[idx_t], ACT[idx_t])
            tloss = F.mse_loss(pp, NXT[idx_t]).item()
            tacc = ((pp[:,:12].round().clamp(0,1) == NXT[idx_t][:,:12]).float().mean().item())*100
            
            # Encode test maze
            states = {}
            for y in range(TEST_MAZE.shape[0]):
                for x in range(TEST_MAZE.shape[1]):
                    if TEST_MAZE[y,x]==CELL_WALL: continue
                    for f in range(4):
                        o = get_obs(TEST_MAZE,(y,x),f)
                        ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                        _, h = model(ot, torch.tensor([[2]], device=DEVICE))
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
        pbar.set_postfix(loss=f'{loss.item():.4f}', tloss=f'{tloss:.4f}',
                         acc=f'{tacc:.0f}%', ratio=f'{ratio:.2f}x')

print(f'Train: {(time.time()-t0)/60:.1f}min')

# ── Final eval ──
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
                _, h = model(ot, torch.tensor([[2]], device=DEVICE))
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
    print(f'  (Baseline without position: 1.00x)')
    
    pca = PCA(n_components=2)
    s2d = pca.fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(8,6))
    colors = [(k[0]/13, k[1]/13, 0.5) for k in keys]
    ax.scatter(s2d[:,0], s2d[:,1], c=colors, s=15, alpha=0.7)
    ax.set_title(f'GRU + position input (ratio={ratio:.2f}x, PCA {pca.explained_variance_ratio_.sum():.0%})')
    plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_posinput.png'), dpi=150)
    torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_posinput.pt'))
    print(f'  Saved.')
