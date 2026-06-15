"""Predict-Compare-Correct: model predicts next state/obs, compares with actual,
and learns to correct its state based on prediction error.

This mirrors how humans maintain a world model:
  predict → act → observe → compare → correct → predict (loop)

Key: prediction error IS the training signal for correction.
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

# ── Predict-Compare-Correct Model ──
class PredCorrModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: obs → state
        self.encoder = nn.Sequential(
            nn.Linear(12, DM), nn.GELU(),
            nn.Linear(DM, DM), nn.GELU(),
            nn.Linear(DM, DM),
        )
        # Transition: (state, action) → pred_next_state
        self.transition = nn.Sequential(
            nn.Linear(DM + 2, DM), nn.GELU(),
            nn.Linear(DM, DM),
        )
        self.act_embed = nn.Embedding(3, 2)
        # Observer: state → pred_obs (for computing prediction error)
        self.observer = nn.Linear(DM, 12)
        # Corrector: (state, pred_error) → corrected_state
        self.corrector = nn.Sequential(
            nn.Linear(DM + 12, DM), nn.GELU(),
            nn.Linear(DM, DM),
        )
        self.corrector_gate = nn.Linear(DM + 12, DM)
        nn.init.zeros_(self.corrector_gate.weight)
        nn.init.zeros_(self.corrector_gate.bias)
        
        for m in [self.encoder, self.transition, self.corrector]:
            for p in m:
                if isinstance(p, nn.Linear): nn.init.xavier_uniform_(p.weight, 0.5)

    def forward(self, obs, act):
        """Predict forward, return everything."""
        state = self.encoder(obs)  # [B, DM]
        act_e = self.act_embed(act)  # [B, 2]
        pred_next = self.transition(torch.cat([state, act_e], -1))  # [B, DM]
        pred_obs = self.observer(pred_next)  # [B, 12]
        return state, pred_next, pred_obs
    
    def correct(self, state, actual_obs):
        """Correct state based on observation."""
        pred_obs = self.observer(state)  # what did we expect to see?
        error = actual_obs - pred_obs  # prediction error
        inp = torch.cat([state, error], -1)
        gate = torch.sigmoid(self.corrector_gate(inp))
        correction = self.corrector(inp)
        return state + gate * correction, pred_obs, error

model = PredCorrModel().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ── Data ──
ALL_MAZES = [generate_maze() for _ in range(20)]

def gen_data(n_trajs=4000, traj_len=20):
    all_obs, all_act, all_next_obs, all_pos, all_next_pos = [], [], [], [], []
    for _ in range(n_trajs):
        maze = random.choice(ALL_MAZES)
        h,w = maze.shape
        pos = (random.randrange(0,h), random.randrange(0,w))
        while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
        facing = random.randint(0,3)
        for _ in range(traj_len):
            o=get_obs(maze,pos,facing)
            all_obs.append(o)
            all_pos.append(pos)
            acts=av_acts(maze,pos,facing)
            act=random.choice(acts)
            if act==2:
                dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
                pos=(ny,nx)
            elif act==0: facing=(facing-1)%4
            else: facing=(facing+1)%4
            no=get_obs(maze,pos,facing)
            all_act.append(act)
            all_next_obs.append(no)
            all_next_pos.append(pos)
    return (torch.tensor(np.array(all_obs),device=DEVICE).float(),
            torch.tensor(all_act,device=DEVICE).long(),
            torch.tensor(np.array(all_next_obs),device=DEVICE).float(),
            torch.tensor(all_pos,device=DEVICE).float(),
            torch.tensor(all_next_pos,device=DEVICE).float())

OBS, ACT, NXT, POS, NPOS = gen_data()
N_SAMPLES = OBS.size(0)
print(f'Data: {N_SAMPLES} transitions')

# ════════════════════════════════════════
# Train
# ════════════════════════════════════════

BSZ, N_STEPS = 256, 8000
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
TEST_MAZE = generate_maze(13, 13)

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    idx = torch.randint(0, N_SAMPLES-BSZ, (BSZ,))
    obs = OBS[idx]; act = ACT[idx]; nxt = NXT[idx]; pos = POS[idx]; npos = NPOS[idx]
    
    state, pred_next, pred_obs = model(obs, act)
    
    # Loss 1: prediction error (predict next obs)
    pred_loss = F.mse_loss(pred_obs, nxt)
    
    # Loss 2: state error (predicted next state should be close to real next state)
    # Real next state = encoder(nxt)
    with torch.no_grad():
        real_next = model.encoder(nxt)
    state_loss = F.mse_loss(pred_next, real_next)
    
    # Loss 3: corrector should reduce error
    corrected, _, _ = model.correct(model.encoder(obs), nxt)
    correct_loss = F.mse_loss(corrected, real_next)
    
    # Loss 4: contrastive - organize state space by position
    state_all = model.encoder(OBS[torch.randint(0, N_SAMPLES-256, (256,))])
    pos_all = POS[torch.randint(0, N_SAMPLES-256, (256,))]
    if state_all.size(0) > 10:
        d = torch.cdist(pos_all, pos_all, p=1)
        target = torch.exp(-d / 3.0)
        h_n = F.normalize(state_all, dim=-1)
        actual = h_n @ h_n.T
        mask = ~torch.eye(state_all.size(0), dtype=torch.bool, device=DEVICE)
        cont_loss = F.mse_loss(actual[mask], target[mask])
    else:
        cont_loss = torch.tensor(0.0, device=DEVICE)
    
    loss = pred_loss + 0.5 * state_loss + 0.3 * correct_loss + 0.2 * cont_loss
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            idx_t = torch.randint(0, N_SAMPLES-256, (256,))
            s, pn, po = model(OBS[idx_t], ACT[idx_t])
            pl = F.mse_loss(po, NXT[idx_t]).item()
            rn = model.encoder(NXT[idx_t])
            sl = F.mse_loss(pn, rn).item()
            co, _, _ = model.correct(model.encoder(OBS[idx_t]), NXT[idx_t])
            cl = F.mse_loss(co, rn).item()
            
            # Encode test maze
            states = {}
            for y in range(TEST_MAZE.shape[0]):
                for x in range(TEST_MAZE.shape[1]):
                    if TEST_MAZE[y,x]==CELL_WALL: continue
                    for f in range(4):
                        o = get_obs(TEST_MAZE,(y,x),f)
                        ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                        st = model.encoder(ot).squeeze(0).cpu().numpy()
                        states[(y,x,f)] = st
            
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
            
            # Navigation test
            nav_correct = 0
            nav_total = 0
            for y in range(1, TEST_MAZE.shape[0]-1):
                for x in range(1, TEST_MAZE.shape[1]-1):
                    if TEST_MAZE[y,x] == CELL_WALL: continue
                    for f in range(4):
                        acts = av_acts(TEST_MAZE, (y,x), f)
                        if 2 not in acts: continue
                        dy,dx = DIRS[f]
                        ny, nx = y+dy, x+dx
                        if not (0<=ny<TEST_MAZE.shape[0] and 0<=nx<TEST_MAZE.shape[1]): continue
                        if TEST_MAZE[ny,nx] == CELL_WALL: continue
                        
                        # Using corrected state to pick action: pick action that brings 
                        # pred_next_state closest to real_next_state (i.e., we know where we'll be)
                        o = get_obs(TEST_MAZE, (y,x), f)
                        ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
                        st = model.encoder(ot)
                        corrected_st, _, _ = model.correct(st, torch.from_numpy(get_obs(TEST_MAZE, (y,x), f)).float().unsqueeze(0).to(DEVICE))
                        ae = model.act_embed(torch.tensor([[2]], device=DEVICE)).squeeze(1)
                        pred_ns = model.transition(torch.cat([corrected_st, ae], -1))
                        
                        # Real next state
                        real_ns = model.encoder(torch.from_numpy(get_obs(TEST_MAZE, (ny,nx), f)).float().unsqueeze(0).to(DEVICE))
                        
                        if (pred_ns - real_ns).norm().item() < (st - real_ns).norm().item():
                            nav_correct += 1
                        nav_total += 1
            
        model.train()
        pbar.set_postfix(pl=f'{pl:.4f}', sl=f'{sl:.4f}', cl=f'{cl:.4f}',
                         ratio=f'{ratio:.2f}x', nav=f'{nav_correct/max(nav_total,1)*100:.0f}%')

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
                # With correction
                st = model.encoder(ot)
                corr, _, _ = model.correct(st, ot)
                states[(y,x,f)] = corr.squeeze(0).cpu().numpy()
    
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
    print(f'  Corrected state ratio: {np.mean(rand_d)/max(np.mean(adj_d),1e-8):.2f}x')
    
    # PCA
    pca = PCA(n_components=2)
    s2d = pca.fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(8,6))
    colors = [(k[0]/13, k[1]/13, 0.5) for k in keys]
    ax.scatter(s2d[:,0], s2d[:,1], c=colors, s=15, alpha=0.7)
    ax.set_title(f'Corrected State Space (PCA {pca.explained_variance_ratio_.sum():.0%})')
    plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_predcorr.png'), dpi=150)
    print('  Saved maze_predcorr.png')

torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_predcorr.pt'))
print('Done.')
