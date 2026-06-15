"""Watch the agent navigate a maze, with live predictions.
"""
import os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DM = 64
CELL_WALL, CELL_PATH, CELL_EXIT = 0, 1, 2
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Maze generator (same as maze_wm.py) ──

def generate_maze(h=15, w=15):
    maze = np.ones((h, w), dtype=np.int32) * CELL_WALL
    start = (1, 1); maze[start] = CELL_PATH
    stack = [start]
    rng = random.Random(42)
    while stack:
        cy, cx = stack[-1]
        nbs = []
        for dy,dx in [(-2,0),(2,0),(0,-2),(0,2)]:
            ny, nx = cy+dy, cx+dx
            if 0<ny<h-1 and 0<nx<w-1 and maze[ny,nx]==CELL_WALL:
                nbs.append((ny,nx))
        if nbs:
            ny, nx = rng.choice(nbs)
            maze[cy+(ny-cy)//2][cx+(nx-cx)//2] = CELL_PATH
            maze[ny,nx] = CELL_PATH; stack.append((ny,nx))
        else: stack.pop()
    maze[h-2,w-2] = CELL_EXIT
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        if maze[h-2+dy,w-2+dx] == CELL_WALL:
            maze[h-2+dy,w-2+dx] = CELL_PATH
    return maze

def get_obs(maze, pos, facing):
    obs = []
    for i in range(4):
        dy,dx = DIRS[(facing+i)%4]; ny,nx = pos[0]+dy, pos[1]+dx
        cell = maze[ny,nx] if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] else CELL_WALL
        o = [0,0,0]; o[cell] = 1; obs.extend(o)
    return np.array(obs, dtype=np.float32)

def available_actions(maze, pos, facing):
    acts = [0, 1]
    dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]!=CELL_WALL:
        acts.append(2)
    return acts

def apply_action(_, pos, facing, action):
    if action == 0: return pos, (facing-1)%4, 0
    if action == 1: return pos, (facing+1)%4, 0
    dy,dx = DIRS[facing]; ny,nx = pos[0]+dy, pos[1]+dx
    if maze[ny,nx] != CELL_WALL:
        return (ny,nx), facing, 1.0 if maze[ny,nx]==CELL_EXIT else 0.0
    return pos, facing, -0.1

# ── Model (same as maze_wm.py) ──

class MazeWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder = nn.Linear(12, DM)
        self.act_embed = nn.Embedding(3, DM//2)
        self.cell = nn.GRUCell(DM + DM//2, DM)
        self.obs_predictor = nn.Linear(DM, 12)
    def forward(self, obs, act, h=None):
        oe = self.obs_encoder(obs); ae = self.act_embed(act)
        inp = torch.cat([oe, ae.squeeze(1)], -1)
        if h is None: h = torch.zeros(obs.size(0), DM, device=DEVICE)
        h = self.cell(inp, h)
        return self.obs_predictor(h), h

model = MazeWM().to(DEVICE)
ckpt = torch.load(os.path.join(BASE_DIR, 'checkpoints', 'maze_wm.pt'), weights_only=False, map_location=DEVICE)
model.load_state_dict(ckpt['model']); model.eval()

# ── Render ──

maze = generate_maze(15, 15)
pos, facing = (1, 1), random.randint(0, 3)
path = [pos]
hist, correct, total = [], 0, 0

print(f'\n{"="*50}')
print(f' Maze size: 15×15  |  Start: {pos}  |  Exit: bottom-right')
print(f' {"="*50}')
print(' Legend:  🤖 = agent  ·· = trail  🚪 = exit')
print(' Model shows POV + predictions for each action.')
print(f' {"="*50}\n')
time.sleep(1)

for step in range(40):
    actions = available_actions(maze, pos, facing)
    act = random.choice(actions)
    act_names = ['← LEFT', '→ RIGHT', '↑ FORWARD']
    
    # Predict
    obs = get_obs(maze, pos, facing)
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
    preds, states = {}, None
    with torch.no_grad():
        for a in [0, 1, 2]:
            if a in actions:
                p, s = model(obs_t, torch.tensor([[a]], device=DEVICE))
                preds[a] = p.squeeze(0).cpu().numpy()
                if a == act: states = s
    
    # Execute
    new_pos, new_facing, reward = apply_action(maze, pos, facing, act)
    actual = get_obs(maze, new_pos, new_facing)
    pos, facing = new_pos, new_facing
    path.append(pos)
    
    # Check prediction
    if act in preds:
        pred_cells = np.argmax(preds[act].reshape(4, 3), axis=1)
        actual_cells = np.argmax(actual.reshape(4, 3), axis=1)
        correct += (pred_cells == actual_cells).sum()
        total += 4
    
    # Render
    os.system('clear')
    cells = {CELL_WALL: '██', CELL_PATH: '  ', CELL_EXIT: '░░'}
    for y in range(maze.shape[0]):
        row = ''
        for x in range(maze.shape[1]):
            if (y,x) == pos: row += '🤖'
            elif (y,x) in path: row += '··'
            else: row += cells[maze[y,x]] + ' '
        if y == 0: row += f'    Step {step+1}/40'
        if y == 1: row += f'    Pos: {pos}  Face: {["↑","→","↓","←"][facing]}'
        if y == 2: row += f'    Action: {act_names[act]}'
        if y == 3: row += f'    Pred avg: {correct/max(total,1)*100:.0f}%'
        if y == 5: row += '    ─── POV ───'
        for i in range(4):
            if y == 6+i:
                dn = ['↑ F', '→ R', '↓ B', '← L'][i]
                if act in preds:
                    pc = ['__','##','XX'][preds[act][i*3:(i+1)*3].argmax()]
                    ac = ['__','##','XX'][actual[i*3:(i+1)*3].argmax()]
                    row += f'    {dn}: pred={pc} actual={ac}'
        if y == 11: row += '    ─── Other actions ───'
        for i2, a2 in enumerate([a for a in actions if a != act]):
            if y == 12+i2:
                an = ['← L','→ R','↑ F'][a2]
                pc = ['__','##','XX'][preds[a2].reshape(4,3).argmax(axis=1)[0]]
                row += f'    {an}: {pc}'
        print(row)
    
    time.sleep(0.15)
    if maze[pos] == CELL_EXIT:
        print(f'\n  🎉 Reached exit!')
        break

print(f'\nDone. {step+1} steps, prediction accuracy: {correct/max(total,1)*100:.0f}%')
