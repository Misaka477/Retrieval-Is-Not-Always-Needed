"""Transformer walks the maze: no state compression, full attention.

If Transformer produces structured states while GRU doesn't,
it proves the bottleneck is the 'compress into single vector' paradigm.
"""
import os, sys, random, time, math
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

# ── Mini Transformer ──
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
        """obs_seq: [B, T, 12] sequence of observations."""
        x = self.obs_embed(obs_seq)  # [B, T, DM]
        x = self.pos_enc(x)
        x = self.transformer(x)  # [B, T, DM]
        x = self.norm(x)
        return self.predictor(x)  # [B, T, 12]

model = MiniTransformer().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ── Data: trajectories ──
ALL_MAZES = [generate_maze() for _ in range(15)]

def gen_trajectories(n_trajs=5000, traj_len=16):
    all_seqs, all_targets = [], []
    for _ in range(n_trajs):
        maze = random.choice(ALL_MAZES)
        h,w = maze.shape
        pos = (random.randrange(0,h), random.randrange(0,w))
        while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
        facing = random.randint(0,3)
        seq = []
        for _ in range(traj_len):
            o = get_obs(maze, pos, facing)
            seq.append(o)
            acts = av_acts(maze, pos, facing)
            act = random.choice(acts)
            if act == 2:
                dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
                pos = (ny,nx)
            elif act == 0: facing = (facing-1)%4
            else: facing = (facing+1)%4
        # Target: predict next observation at each position
        all_seqs.append(seq[:-1])
        all_targets.append(seq[1:])
    return (torch.tensor(np.array(all_seqs), device=DEVICE).float(),
            torch.tensor(np.array(all_targets), device=DEVICE).float())

SEQ, TGT = gen_trajectories(5000, 16)
N_SAMPLES = SEQ.size(0)
print(f'Data: {N_SAMPLES} trajectories × {SEQ.size(1)} steps')

# ════════════════════════════════════════
# Train
# ════════════════════════════════════════

BSZ, N_STEPS = 128, 5000
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

TEST_MAZE = generate_maze(13, 13)

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    idx = torch.randint(0, N_SAMPLES-BSZ, (BSZ,))
    seq = SEQ[idx]; target = TGT[idx]
    
    pred = model(seq)
    loss = F.mse_loss(pred, target)
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            idx_t = torch.randint(0, N_SAMPLES-32, (32,))
            pp = model(SEQ[idx_t])
            tloss = F.mse_loss(pp, TGT[idx_t]).item()
            tacc = (pp.round().clamp(0,1) == TGT[idx_t]).float().mean().item()
            
            # Encode test maze positions using a single-step trajectory
            # Use the last hidden state of the transformer as the "state"
            states = {}
            with torch.no_grad():
                for y in range(TEST_MAZE.shape[0]):
                    for x in range(TEST_MAZE.shape[1]):
                        if TEST_MAZE[y,x]==CELL_WALL: continue
                        for f in range(4):
                            o = get_obs(TEST_MAZE,(y,x),f)
                            # Create a 1-step sequence
                            ot = torch.from_numpy(o).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                            p = model(ot)  # predict next obs
                            # Get the embedding before the predictor
                            emb = model.obs_embed(ot)
                            emb = model.pos_enc(emb)
                            # After first transformer layer
                            h = model.transformer.layers[0](emb)
                            states[(y,x,f)] = h[0,-1].cpu().numpy()
            
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
                         acc=f'{tacc*100:.0f}%', ratio=f'{ratio:.2f}x')

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
                ot = torch.from_numpy(o).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                emb = model.pos_enc(model.obs_embed(ot))
                h = model.transformer.layers[0](emb)
                states[(y,x,f)] = h[0,-1].cpu().numpy()
    
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
    print(f'  Transformer state ratio: {ratio:.2f}x')
    print(f'  (GRU ratio: ~1.00x)')
    
    # PCA
    pca = PCA(n_components=2)
    s2d = pca.fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(8,6))
    colors = [(k[0]/13, k[1]/13, 0.5) for k in keys]
    ax.scatter(s2d[:,0], s2d[:,1], c=colors, s=15, alpha=0.7)
    ax.set_title(f'Transformer State Space (PCA {pca.explained_variance_ratio_.sum():.0%})')
    plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_transformer.png'), dpi=150)
    print('  Saved maze_transformer.png')

torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_transformer.pt'))
print('Done.')
