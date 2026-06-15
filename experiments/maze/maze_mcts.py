"""AlphaGo式 MCTS + 策略-价值网络 走迷宫。
在脑子里模拟N步再决定，通过搜索解决稀疏奖励问题。"""
import os, sys, random, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DM = 64
CELL_WALL, CELL_PATH, CELL_EXIT = 0, 1, 2
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_DIM = 14

def generate_maze(h=9, w=9):
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
    h,w=maze.shape; obs.append(pos[0]/h); obs.append(pos[1]/w)
    return np.array(obs)

def valid_acts(maze, pos, facing):
    acts=[0,1]
    dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL: acts.append(2)
    return acts

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, DM), nn.ReLU(),
            nn.Linear(DM, DM), nn.ReLU(),
        )
        self.policy = nn.Linear(DM, 3)
        self.value = nn.Linear(DM, 1)
    def forward(self, obs):
        h = self.net(obs)
        return self.policy(h), self.value(h)

# ═══════════════════════════════════════
# MCTS Node
# ═══════════════════════════════════════

class MCTSNode:
    def __init__(self, maze, pos, facing, prev_action=None):
        self.maze = maze
        self.pos = pos
        self.facing = facing
        self.prev_action = prev_action  # how we got here
        self.valid_acts = valid_acts(maze, pos, facing)
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = {}  # prior policy probabilities
    
    def is_terminal(self):
        return self.maze[self.pos] == CELL_EXIT
    
    def best_child(self, c_puct=2.0):
        """UCB: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)"""
        best_a, best_score = None, -1e9
        for a in self.valid_acts:
            child = self.children.get(a)
            if child is None:
                score = c_puct * self.prior.get(a, 0.5) * math.sqrt(self.visits + 1) / 1
            else:
                q = child.value_sum / (child.visits + 1e-8)
                score = q + c_puct * self.prior.get(a, 0.5) * math.sqrt(self.visits) / (1 + child.visits)
            if score > best_score: best_a, best_score = a, score
        return best_a

class MCTS:
    def __init__(self, model, n_sim=100, c_puct=2.0, gamma=0.99):
        self.model = model
        self.n_sim = n_sim
        self.c_puct = c_puct
        self.gamma = gamma
    
    def simulate(self, node):
        """One MCTS simulation: select → expand → rollout → backup"""
        path = [node]
        
        # SELECT: descent to leaf
        while node.children and not node.is_terminal():
            a = node.best_child(self.c_puct)
            if a not in node.children:
                break  # unexpanded child
            node = node.children[a]
            path.append(node)
        
        leaf = path[-1]
        
        if leaf.is_terminal():
            value = 100.0
        else:
            # EXPAND: add all children
            for a in leaf.valid_acts:
                if a not in leaf.children:
                    # Compute next state
                    pos, facing = leaf.pos, leaf.facing
                    if a == 2:
                        dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
                        if 0<=ny<leaf.maze.shape[0] and 0<=nx<leaf.maze.shape[1] and leaf.maze[ny,nx]!=CELL_WALL:
                            pos = (ny, nx)
                    elif a == 0: facing = (facing-1)%4
                    else: facing = (facing+1)%4
                    leaf.children[a] = MCTSNode(leaf.maze, pos, facing, a)
            
            # Value estimation from neural network only (no rollout)
            obs = get_obs(leaf.maze, leaf.pos, leaf.facing)
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                policy_logits, value = self.model(ot)
                value = value.item()
            
            # Store policy priors
            probs = F.softmax(policy_logits, dim=-1).squeeze(0)
            for a in leaf.valid_acts:
                leaf.prior[a] = probs[a].item()
            
            # Leaf value = neural net value (clamped)
            value = max(-50, min(100, value))
        
        # BACKUP: propagate value up the path
        discount = 1.0
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value * discount
            discount *= self.gamma
    
    def search(self, root_node):
        """Run n_sim simulations from root, return action visit distribution."""
        for _ in range(self.n_sim):
            self.simulate(root_node)
        
        # Visit counts for each valid action
        visits = {a: 0 for a in root_node.valid_acts}
        for a, child in root_node.children.items():
            visits[a] = child.visits
        return visits

# ═══════════════════════════════════════
# Training + Live Demo
# ═══════════════════════════════════════

model = PolicyNet().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
mcts = MCTS(model, n_sim=20, c_puct=1.0)

ALL_MAZES = [generate_maze(9, 9) for _ in range(30)]
DEMO = generate_maze(9, 9)
DEMO_EXIT = (7, 7)

plt.ion()
fig, (ax_maze, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('MCTS Maze - Think before you walk', fontsize=12, fontweight='bold')
cmap = ListedColormap(['#2d2d2d', '#f0f0f0', '#4ecdc4'])
reward_hist = []
success_hist = []
t0 = time.time()

N_EPISODES = 20000
UPD = 100

for ep in range(N_EPISODES):
    maze = random.choice(ALL_MAZES)
    h,w = maze.shape; EXIT = (h-2, w-2)
    pos = (random.randrange(0,h), random.randrange(0,w))
    while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
    facing = random.randint(0,3)
    
    ep_obs, ep_acts, ep_targets = [], [], []
    total_reward = 0
    root = MCTSNode(maze, pos, facing)
    
    for _ in range(100):
        visits = mcts.search(root)
        
        # Best action = most visited
        best_a = max(visits, key=visits.get) if visits else root.valid_acts[0]
        
        # Training target: MCTS visit distribution
        total_visits = sum(visits.values())
        if total_visits == 0:
            target_probs = {a: 1.0/len(visits) for a in visits}
        else:
            target_probs = {a: v/total_visits for a,v in visits.items()}
        
        ep_obs.append(get_obs(maze, pos, facing))
        ep_acts.append(best_a)
        ep_targets.append(target_probs)
        
        # Execute best action
        if best_a == 2:
            dy,dx = DIRS[facing]; ny, nx = pos[0]+dy, pos[1]+dx
            if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL: pos=(ny,nx)
        elif best_a == 0: facing = (facing-1)%4
        else: facing = (facing+1)%4
        
        dist = abs(pos[0]-EXIT[0])+abs(pos[1]-EXIT[1])
        reward = 100 if maze[pos]==CELL_EXIT else -dist*0.1 - 0.5
        total_reward += reward
        
        if maze[pos]==CELL_EXIT or best_a not in root.children:
            break
        
        root = root.children[best_a]
    
    # Train on episode data
    if len(ep_obs) >= 2:
        obs_t = torch.from_numpy(np.array(ep_obs)).float().to(DEVICE)
        acts_t = torch.tensor(ep_acts, device=DEVICE)
        target_probs_t = torch.zeros(len(ep_obs), 3, device=DEVICE)
        for i, t in enumerate(ep_targets):
            for a, p in t.items():
                target_probs_t[i, a] = p
        
        logits, values = model(obs_t)
        
        # Policy loss: cross-entropy with MCTS visit distribution
        policy_loss = -(target_probs_t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        
        # Value: discounted return
        vs = torch.zeros(len(ep_obs), device=DEVICE)
        G = 100.0 if maze[pos]==CELL_EXIT else -float(abs(pos[0]-EXIT[0])+abs(pos[1]-EXIT[1]))*0.1
        for i in range(len(ep_obs)-1, -1, -1):
            d = abs(EXIT[0]-pos[0])+abs(EXIT[1]-pos[1])  # approx
            G = G * 0.99 - 0.2
            vs[i] = G
        
        value_loss = F.mse_loss(values.squeeze(), vs)
        loss = policy_loss + 0.5 * value_loss
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    
    reward_hist.append(total_reward)
    success_hist.append(maze[pos]==CELL_EXIT)
    
    # Show every episode on demo maze
    succ_rate = np.mean(success_hist[-500:])*100
    avg_r = np.mean(reward_hist[-100:])
    
    ax_maze.clear()
    ax_maze.imshow(DEMO, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    ax_maze.set_xticks([]); ax_maze.set_yticks([])
    
    # Walk demo maze with MCTS (not raw policy - avoids random stuck)
    ppos, pfacing = (1,1), 0
    ppath = [(1,1)]
    model.eval()
    for _ in range(30):
        root = MCTSNode(DEMO, ppos, pfacing)
        visits = mcts.search(root)
        total_v = sum(visits.values())
        if total_v == 0:
            a = random.choice(root.valid_acts)
        else:
            # Sample from visit distribution (adds exploration)
            acts, weights = zip(*visits.items())
            a = random.choices(acts, weights=weights, k=1)[0]
        
        if a == 2:
            dy,dx=DIRS[pfacing]; ny,nx=ppos[0]+dy,ppos[1]+dx
            if 0<=ny<DEMO.shape[0] and 0<=nx<DEMO.shape[1] and DEMO[ny,nx]!=CELL_WALL: ppos=(ny,nx)
        elif a == 0: pfacing=(pfacing-1)%4
        else: pfacing=(pfacing+1)%4
        ppath.append(ppos)
        
        ax_maze.clear()
        ax_maze.imshow(DEMO, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        pa = np.array(ppath)
        ax_maze.plot(pa[:,1], pa[:,0], '-', color='#74b9ff', alpha=0.4, linewidth=1.5)
        ax_maze.plot(ppos[1], ppos[0], 'o', markersize=10, color='#ff6b6b', zorder=5)
        ex_ = np.where(DEMO==CELL_EXIT)
        if len(ex_[0]): ax_maze.plot(ex_[1][0], ex_[0][0], '*', markersize=16, color='#ffd93d', zorder=4)
        ax_maze.set_title(f'Ep {ep} | Succ {succ_rate:.0f}% | AvgR {avg_r:.1f}')
        ax_plot.clear()
        if len(reward_hist)>50:
            c = np.convolve(reward_hist, np.ones(50)/50, mode='valid')
            ax_plot.plot(c, linewidth=1, color='#0984e3')
        ax_plot.set_title('Reward (50-ep avg)')
        ax_plot.grid(True, alpha=0.2)
        plt.pause(0.02)
        if DEMO[ppos]==CELL_EXIT: break
    model.train()

plt.ioff()
print(f'Done. Success rate (last 500): {np.mean(success_hist[-500:])*100:.0f}%')
plt.show()
