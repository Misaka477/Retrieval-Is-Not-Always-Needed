"""迷宫强化学习：模型自己走，走到出口有奖，撞墙罚。

无预测、无MSE、无CE。只有策略梯度。
学会找出口 = 状态结构自然带空间信息。
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
OBS_DIM = 12 + 2  # 格+位置
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
    h,w=maze.shape; obs.append(pos[0]/h); obs.append(pos[1]/w)
    return np.array(obs,dtype=np.float32)

def av_acts(maze, pos, facing):
    acts=[0,1]
    dy,dx=DIRS[facing]; ny,nx=pos[0]+dy,pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL: acts.append(2)
    return acts

# ── 纯强化学习策略网络 ──
class Policy(nn.Module):
    """观测 → 动作概率 + 价值估计。没有预测头，没有MSE。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, DM),
            nn.ReLU(),
            nn.Linear(DM, DM),
            nn.ReLU(),
        )
        self.policy = nn.Linear(DM, 3)  # 动作 logits
        self.value = nn.Linear(DM, 1)   # 状态价值 V(s)
    
    def forward(self, obs):
        h = self.net(obs)
        logits = self.policy(h)
        v = self.value(h)
        return logits, v
    
    def act(self, obs, valid_acts, eps=0.1):
        """选动作（ɛ-贪心）"""
        logits, v = self.forward(obs)
        # 掩码无效动作
        mask = torch.full((3,), -1e9, device=DEVICE)
        for a in valid_acts: mask[a] = 0
        logits = logits + mask
        
        probs = F.softmax(logits, dim=-1)
        a = torch.multinomial(probs, 1).item() if random.random() >= eps else random.choice(valid_acts)
        
        log_prob = F.log_softmax(logits, dim=-1)[0, a]
        return a, log_prob, v

model = Policy().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ── 强化学习训练 ──
ALL_MAZES = [generate_maze() for _ in range(30)]
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
MAX_STEPS = 100
GAMMA = 0.99

n_episodes = 5000
reward_history = []
step_history = []

pbar = tqdm(range(n_episodes))
t0 = time.time()

for ep in pbar:
    maze = random.choice(ALL_MAZES)
    h,w = maze.shape
    EXIT = (h-2, w-2)
    
    # 随机起点
    pos = (random.randrange(0,h), random.randrange(0,w))
    while maze[pos]==CELL_WALL: pos=(random.randrange(0,h), random.randrange(0,w))
    facing = random.randint(0,3)
    
    log_probs = []
    values = []
    rewards = []
    
    for step in range(MAX_STEPS):
        obs = get_obs(maze, pos, facing)
        ot = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        valid = av_acts(maze, pos, facing)
        
        old_dist = abs(pos[0]-EXIT[0]) + abs(pos[1]-EXIT[1])
        a, lp, v = model.act(ot, valid, eps=max(0.05, 0.5*(1-ep/n_episodes)))
        
        # 执行动作
        if a == 2:
            dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
            if 0<=ny<h and 0<=nx<w and maze[ny,nx]!=CELL_WALL:
                pos = (ny,nx)
        elif a == 0: facing = (facing-1)%4
        else: facing = (facing+1)%4
        
        # 奖励
        if maze[pos] == CELL_EXIT:
            reward = 100.0
        else:
            new_dist = abs(pos[0]-EXIT[0]) + abs(pos[1]-EXIT[1])
            dist_progress = old_dist - new_dist  # 正=靠近出口
            reward = dist_progress * 2 - 0.5  # 靠近奖励，远离惩罚，每步烧0.5
        
        log_probs.append(lp)
        values.append(v)
        rewards.append(reward)
        
        if maze[pos] == CELL_EXIT:
            break
    
    # 计算回报并更新策略
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, device=DEVICE)
    values_t = torch.cat(values).squeeze()
    
    # 优势 = 回报 - 价值估计
    advantages = returns - values_t.detach()
    
    # 策略梯度损失
    log_probs_t = torch.stack(log_probs)
    policy_loss = -(log_probs_t * advantages).mean()
    
    # 价值损失
    value_loss = F.mse_loss(values_t, returns)
    
    loss = policy_loss + 0.5 * value_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    
    total_reward = sum(rewards)
    reward_history.append(total_reward)
    step_history.append(step+1)
    
    if ep % 500 == 0:
        avg_r = np.mean(reward_history[-100:]) if reward_history else 0
        avg_s = np.mean(step_history[-100:]) if step_history else 0
        pbar.set_postfix(avg_reward=f'{avg_r:.1f}', avg_steps=f'{avg_s:.1f}')
        torch.save({'model': model.state_dict(), 'ep': ep},
                   os.path.join(BASE_DIR, 'checkpoints', f'maze_rl_{ep}.pt'))

print(f'Train: {(time.time()-t0)/60:.1f}min')

# ── 最终评估 ──
print('\n=== 评估 ===')
TEST_MAZE = generate_maze(13, 13)
EXIT = (11, 11)
n_tests = 100
success = 0
steps_total = 0

for t in range(n_tests):
    pos = (1, 1)
    facing = 0
    for s in range(200):
        obs = get_obs(TEST_MAZE, pos, facing)
        ot = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        valid = av_acts(TEST_MAZE, pos, facing)
        logits, _ = model.forward(ot)
        mask = torch.full((3,), -1e9, device=DEVICE)
        for a in valid: mask[a] = 0
        a = (logits + mask).argmax().item()
        
        if a == 2:
            dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
            if 0<=ny<TEST_MAZE.shape[0] and 0<=nx<TEST_MAZE.shape[1] and TEST_MAZE[ny,nx]!=CELL_WALL:
                pos = (ny,nx)
        elif a == 0: facing = (facing-1)%4
        else: facing = (facing+1)%4
        
        if TEST_MAZE[pos] == CELL_EXIT:
            success += 1
            steps_total += s+1
            break

print(f'  {n_tests} 轮测试: {success} 次成功 ({success/n_tests*100:.0f}%)')
print(f'  平均步数(成功时): {steps_total/max(success,1):.1f}')

# 状态空间分析
states = {}
for y in range(TEST_MAZE.shape[0]):
    for x in range(TEST_MAZE.shape[1]):
        if TEST_MAZE[y,x]==CELL_WALL: continue
        for f in range(4):
            o = get_obs(TEST_MAZE,(y,x),f)
            ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
            h = model.net(ot).detach().squeeze(0).cpu().numpy()
            states[(y,x,f)] = h

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
print(f'  状态结构比: {ratio:.2f}x')

# 价值函数 vs 出口距离
val_preds = []
for k in keys:
    o = get_obs(TEST_MAZE, (k[0],k[1]), k[2])
    ot = torch.from_numpy(o).float().unsqueeze(0).to(DEVICE)
    _, v = model.forward(ot)
    val_preds.append(v.item())
exit_dists = [abs(k[0]-11)+abs(k[1]-11) for k in keys]
val_corr = np.corrcoef(val_preds, exit_dists)[0,1] if len(exit_dists)>1 else 0
print(f'  V(s)↔出口距离相关: {val_corr:.2f}')

# 图
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# 训练曲线
axes[0].plot(np.convolve(reward_history, np.ones(50)/50, mode='valid'), linewidth=1)
axes[0].set_title(f'训练奖励(滑动50轮平均)\n最终成功率 {success/n_tests*100:.0f}%')
# 状态空间
pca = PCA(n_components=2)
s2d = pca.fit_transform(vecs)
axes[1].scatter(s2d[:,0], s2d[:,1], c=exit_dists, cmap='RdYlGn_r', s=10, alpha=0.6)
axes[1].set_title(f'状态空间(按出口距离着色)\nratio={ratio:.2f}x')
# 价值函数
axes[2].scatter(exit_dists, val_preds, s=5, alpha=0.5)
axes[2].set_xlabel('出口距离'); axes[2].set_ylabel('V(s)')
axes[2].set_title(f'V(s) vs 出口距离 (corr={val_corr:.2f})')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'experiments', 'maze_rl_trained.png'), dpi=150)
torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR, 'checkpoints', 'maze_rl_trained.pt'))
print('Done.')
