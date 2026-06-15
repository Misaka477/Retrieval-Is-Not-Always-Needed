"""Minigrid PPO with proper vec env setup"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import minigrid

# Wrap manually to fix obs space
from gymnasium.wrappers import ReshapeObservation
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

def make_env():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env)  # 84x84x3 image
    env = ImgObsWrapper(env)     # flatten
    return env

env = make_vec_env(make_env, n_envs=4, seed=42)
model = PPO("CnnPolicy", env, verbose=1, n_steps=256, batch_size=128,
            learning_rate=3e-4, n_epochs=10)
model.learn(total_timesteps=200000)

# Test
env_test = make_env()
for ep in range(5):
    obs, _ = env_test.reset()
    total = 0
    for s in range(100):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env_test.step(a)
        total += r
        if term or trunc:
            print(f"Ep {ep}: {s+1} steps, reward={total:.1f}")
            break
