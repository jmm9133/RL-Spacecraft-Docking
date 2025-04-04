# src/rllib_satellite_wrapper.py
import gymnasium as gym
import numpy as np # Make sure numpy is imported
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .satellite_marl_env import raw_env as satellite_pettingzoo_creator
import logging # Import logging

logger = logging.getLogger(__name__)

# --- Define Reward Clipping Range ---
REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX = 10.0
# ----------------------------------

class RllibSatelliteEnv(MultiAgentEnv):
    """
    Wraps the PettingZoo ParallelEnv ('SatelliteMARLEnv') for compatibility
    with the Ray RLlib MultiAgentEnv interface. Adds reward clipping.
    """
    def __init__(self, env_config_dict=None):
        if env_config_dict is None: env_config_dict = {}
        env_config_dict["render_mode"] = None
        self.env = satellite_pettingzoo_creator(**env_config_dict)

        self._agent_ids = set(self.env.possible_agents)
        self.observation_space = gym.spaces.Dict({
            agent: self.env.observation_space(agent)
            for agent in self.env.possible_agents
        })
        self.action_space = gym.spaces.Dict({
            agent: self.env.action_space(agent)
            for agent in self.env.possible_agents
        })

        super().__init__()
        # self._skip_env_checking = True # Usually not needed if step/reset are correct

    def reset(self, *, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        # Ensure observations are finite after reset
        for agent_id, obs in observations.items():
            if not np.all(np.isfinite(obs)):
                logger.warning(f"NaN/Inf in reset obs for {agent_id}. Clamping.")
                observations[agent_id] = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return observations, infos

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)

        # --- FIX: Clip Rewards ---
        clipped_rewards = {}
        for agent_id, reward in rewards.items():
            original_reward = reward # Keep for info if needed
            # Check for non-finite rewards before clipping
            if not np.isfinite(reward):
                logger.warning(f"Non-finite reward {reward} for agent {agent_id}. Clipping to 0.")
                reward = 0.0
            clipped_reward = np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)
            clipped_rewards[agent_id] = clipped_reward
            # Optionally add original reward to info
            # if agent_id in infos: infos[agent_id]['original_reward'] = original_reward
            # else: infos[agent_id] = {'original_reward': original_reward}
        # -------------------------

        # Add __all__ keys
        terminations["__all__"] = all(terminations.get(agent, False) for agent in self.possible_agents)
        truncations["__all__"] = all(truncations.get(agent, False) for agent in self.possible_agents)

        # Ensure observations are finite after step
        for agent_id, obs in observations.items():
            if not np.all(np.isfinite(obs)):
                logger.warning(f"NaN/Inf in step obs for {agent_id}. Clamping.")
                observations[agent_id] = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)


        # Return clipped rewards
        return observations, clipped_rewards, terminations, truncations, infos

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def possible_agents(self):
        return self.env.possible_agents