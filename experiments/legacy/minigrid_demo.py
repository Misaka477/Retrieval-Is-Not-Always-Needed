"""Simple Minigrid demo: train a basic agent step by step, watch it."""
from minigrid.wrappers import ImgObsWrapper
import gymnasium as gym
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

env = ImgObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array"))
obs, _ = env.reset()
obs_dim = obs.flatten().shape[0]
n_acts = env.action_space.n

model = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_acts))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
rew_log = []

for ep in range(500):
    obs, _ = env.reset()
    total_rew = 0
    for step in range(50):
        o = torch.from_numpy(obs.flatten()).float().unsqueeze(0)
        logits = model(o)
        a = torch.multinomial(F.softmax(logits/0.5, -1), 1).item()
        nobs, r, term, trunc, _ = env.step(a)
        total_rew += r
        
        # Simple REINFORCE
        with torch.no_grad():
            baseline = total_rew / max(step+1, 1)
        loss = -F.log_softmax(logits, -1)[0, a] * (r - baseline)
        opt.zero_grad(); loss.backward(); opt.step()
        
        obs = nobs
        if term or trunc: break
    
    rew_log.append(total_rew)
    
    if ep % 20 == 0:
        obs2, _ = env.reset()
        for s2 in range(30):
            o2 = torch.from_numpy(obs2.flatten()).float().unsqueeze(0)
            a2 = model(o2).argmax().item()
            obs2, _, term2, _, _ = env.step(a2)
            ax.clear(); ax.imshow(env.render()); ax.set_title(f"Ep {ep}, step {s2}")
            plt.pause(0.08)
            if term2: break

plt.ioff()
