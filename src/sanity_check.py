# tests/test_rewards_sanity.py
from src.satellite_marl_env import raw_env
import numpy as np

env = raw_env()
obs, infos = env.reset()
for step in range(5000):
    # random actions in [-1,1]^6 for each active agent
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, dones, truncs, infos = env.step(actions)
    print(step, rewards)  # <--- nothing here should ever print nan or inf
    if dones["__all__"] or truncs["__all__"]:
        print("Episode ended at step", step)
        print("Final episode reward:", env.episode_rewards)
        break
