# src/rllib_satellite_wrapper.py
import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from . import config as env_config # Ensure config is imported
from .satellite_marl_env import raw_env as satellite_pettingzoo_creator
import logging

logger = logging.getLogger(__name__)

# --- Reward Clipping Range (Keep as is or adjust if needed) ---
REWARD_CLIP_MIN = -10000.0
REWARD_CLIP_MAX = env_config.REWARD_DOCKING_SUCCESS + 100

# --- Observation Clipping Range (Tighter) ---
# Base clipping on OOB distance plus some margin, maybe velocity limits?
# Example: Clip positions based on OOB distance * 1.5
# Velocities might need different bounds if they can realistically get very high.
OBS_POS_BOUND = env_config.OUT_OF_BOUNDS_DISTANCE * 1.5 # e.g., 15.0
OBS_VEL_BOUND = 10.0 # Example velocity bound (adjust based on expected speeds)
OBS_OTHER_BOUND = 10.0 # Bound for quat components (already normalized) and ang_vel

# Define separate bounds for different parts of the observation vector if needed
# Assuming obs structure: [rel_pos(3), rel_vel(3), own_quat(4), own_ang_vel(3)]
OBS_CLIP_LOW = np.array(
    [-OBS_POS_BOUND]*3 + [-OBS_VEL_BOUND]*3 + [-OBS_OTHER_BOUND]*4 + [-OBS_OTHER_BOUND]*3+[OBS_OTHER_BOUND],
    dtype=np.float32
)
OBS_CLIP_HIGH = np.array(
    [OBS_POS_BOUND]*3 + [OBS_VEL_BOUND]*3 + [OBS_OTHER_BOUND]*4 + [OBS_OTHER_BOUND]*3+[OBS_OTHER_BOUND],
    dtype=np.float32
)
# Make sure the bounds match the observation dimension
expected_obs_dim = env_config.OBS_DIM_PER_AGENT
if OBS_CLIP_LOW.shape[0] != expected_obs_dim or OBS_CLIP_HIGH.shape[0] != expected_obs_dim:
    logger.error(f"FATAL: Observation clipping bounds shape mismatch! Expected {expected_obs_dim}, Got Low={OBS_CLIP_LOW.shape}, High={OBS_CLIP_HIGH.shape}. Check OBS structure assumption.")
    # Fallback to simple scalar clipping if shape is wrong
    OBS_CLIP_LOW = -100.0 # Simpler fallback bound
    OBS_CLIP_HIGH = 100.0
    logger.warning(f"Using fallback scalar observation clipping: [{OBS_CLIP_LOW}, {OBS_CLIP_HIGH}]")


class RllibSatelliteEnv(MultiAgentEnv):
    """
    Wraps the PettingZoo ParallelEnv ('SatelliteMARLEnv') for compatibility
    with the Ray RLlib MultiAgentEnv interface. Adds reward and observation
    clipping for robustness.
    """
    def __init__(self, env_config_dict=None):
        # ... (init remains the same)
        if env_config_dict is None: env_config_dict = {}
        render_mode = env_config_dict.get("render_mode", None)
        self.env = satellite_pettingzoo_creator(render_mode=render_mode, **env_config_dict)
        self._agent_ids = set(self.env.possible_agents)
        self.observation_space = gym.spaces.Dict({
            agent: self.env.observation_space(agent) for agent in self.env.possible_agents
        })
        self.action_space = gym.spaces.Dict({
            agent: self.env.action_space(agent) for agent in self.env.possible_agents
        })
        super().__init__()
        logger.info(f"RllibSatelliteEnv wrapper initialized for agents: {self._agent_ids}")

    def reset(self, *, seed=None, options=None):
        logger.debug("Wrapper: Calling underlying env.reset()")
        observations, infos = self.env.reset(seed=seed, options=options)
        logger.debug(f"Wrapper: Raw reset observations keys: {list(observations.keys())}")
        processed_observations = {}
        for agent_id in self._agent_ids:
            obs = observations.get(agent_id)
            if obs is None:
                logger.error(f"Agent {agent_id} missing from raw reset obs! Zero obs provided.")
                processed_observations[agent_id] = np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
                continue
            if not isinstance(obs, np.ndarray):
                logger.error(f"Reset obs for {agent_id} not numpy array: {type(obs)}. Zero obs provided.")
                processed_observations[agent_id] = np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
                continue
            # Ensure correct shape before clipping
            if obs.shape != self.observation_space[agent_id].shape:
                 logger.error(f"Reset obs shape mismatch for {agent_id}. Expected {self.observation_space[agent_id].shape}, Got {obs.shape}. Zero obs provided.")
                 processed_observations[agent_id] = np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
                 continue

            # NaN/Inf check FIRST
            if not np.all(np.isfinite(obs)):
                logger.warning(f"NaN/Inf in raw reset obs for {agent_id}. Clamping with nan_to_num.")
                # Use clipping bounds when replacing inf
                obs = np.nan_to_num(obs, nan=0.0, posinf=np.inf, neginf=-np.inf) # Replace nan first
                obs = np.clip(obs, OBS_CLIP_LOW, OBS_CLIP_HIGH) # Then clip infs that became large numbers

            # Clip values SECOND using defined bounds
            # original_obs_before_clip = obs.copy() # For logging
            # clipped_obs = np.clip(obs, OBS_CLIP_LOW, OBS_CLIP_HIGH)
            # if not np.array_equal(clipped_obs, original_obs_before_clip):
            #      max_orig = np.max(original_obs_before_clip)
            #      min_orig = np.min(original_obs_before_clip)
            #      max_clip = np.max(clipped_obs)
            #      min_clip = np.min(clipped_obs)
            #      logger.debug(f"Reset obs for {agent_id} clipped. Orig Max/Min: {max_orig:.2f}/{min_orig:.2f}, Clipped Max/Min: {max_clip:.2f}/{min_clip:.2f}")

            # processed_observations[agent_id] = clipped_obs.astype(np.float32)
            processed_observations[agent_id] = obs.astype(np.float32)

        logger.debug(f"Wrapper: Processed reset observations keys: {list(processed_observations.keys())}")
        return processed_observations, infos


    def step(self, action_dict):
        if not hasattr(self, 'step_count'): self.step_count = 0
        self.step_count += 1
        log_frequency = 200 # Log rewards less frequently once stable

        logger.debug(f"Wrapper: Calling underlying env.step() [Internal Step {self.step_count}]")
        try:
             observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
             # logger.debug(f"Wrapper: Raw step results - Obs keys: {list(observations.keys())}, Reward keys: {list(rewards.keys())}")
        except Exception as e:
             logger.exception(f"CRITICAL ERROR in underlying env.step() [Internal Step {self.step_count}]: {e}. Terminating.")
             # ... (dummy return logic remains the same) ...
             dummy_obs = {aid: np.zeros(self.observation_space[aid].shape, dtype=np.float32) for aid in self._agent_ids}
             dummy_rewards = {aid: -500.0 for aid in self._agent_ids}
             dummy_terminations = {aid: True for aid in self._agent_ids}; dummy_terminations["__all__"] = True
             dummy_truncations = {aid: False for aid in self._agent_ids}; dummy_truncations["__all__"] = False
             dummy_infos = {aid: {"error": "underlying env step failed", "exception": str(e)} for aid in self._agent_ids}
             return dummy_obs, dummy_rewards, dummy_terminations, dummy_truncations, dummy_infos

        # --- Process Rewards (Keep previous robust logic) ---
        processed_rewards = {}
        raw_rewards_log = {}
        any_bad_raw_reward = False
        for agent_id in self._agent_ids:
            raw_reward = rewards.get(agent_id)
            raw_rewards_log[agent_id] = raw_reward
            if raw_reward is None:
                processed_reward = 0.0
                any_bad_raw_reward = True
            elif not np.isfinite(raw_reward):
                logger.error(f"!!! Non-finite reward {raw_reward} from raw env for {agent_id} [Step {self.step_count}]! Clipping to zero. !!!")
                processed_reward = 0.0
                any_bad_raw_reward = True
            else:
                processed_reward = np.clip(float(raw_reward), REWARD_CLIP_MIN, REWARD_CLIP_MAX)
            processed_rewards[agent_id] = processed_reward
            # if agent_id not in infos: infos[agent_id] = {} # Ensure info dict exists

        # Log processed rewards
        if (self.step_count % log_frequency == 1) or any_bad_raw_reward:
            rewards_str = ", ".join([f"{k}: {v:.3f}" for k, v in processed_rewards.items()])
            raw_rewards_str = ", ".join([f"{k}: {v}" for k, v in raw_rewards_log.items()])
            logger.info(f"Wrapper Step {self.step_count}: Processed Rewards: {{{rewards_str}}} (Raw: {{{raw_rewards_str}}})")


        # --- Process Observations (with tighter clipping) ---
        processed_observations = {}
        for agent_id in self._agent_ids:
             obs = observations.get(agent_id)
             if obs is None:
                  logger.error(f"Agent {agent_id} missing from raw step obs! Zero obs provided.")
                  processed_observations[agent_id] = np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
                  continue
             if not isinstance(obs, np.ndarray):
                  logger.error(f"Step obs for {agent_id} not numpy array: {type(obs)}. Using zeros.")
                  processed_observations[agent_id] = np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
                  continue
             # Ensure correct shape before clipping
             if obs.shape != self.observation_space[agent_id].shape:
                  logger.error(f"Step obs shape mismatch for {agent_id}. Expected {self.observation_space[agent_id].shape}, Got {obs.shape}. Zero obs provided.")
                  processed_observations[agent_id] = np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
                  continue

             # NaN/Inf check FIRST
             if not np.all(np.isfinite(obs)):
                  logger.warning(f"NaN/Inf in raw step obs for {agent_id} [Step {self.step_count}]. Clamping.")
                  obs = np.nan_to_num(obs, nan=0.0, posinf=np.inf, neginf=-np.inf) # Replace nan first
                  obs = np.clip(obs, OBS_CLIP_LOW, OBS_CLIP_HIGH) # Then clip infs

             # Clip values SECOND
            #  original_obs_before_clip = obs.copy()
            #  clipped_obs = np.clip(obs, OBS_CLIP_LOW, OBS_CLIP_HIGH)
            #  if not np.array_equal(clipped_obs, original_obs_before_clip):
            #        max_orig = np.max(original_obs_before_clip); min_orig = np.min(original_obs_before_clip)
            #        max_clip = np.max(clipped_obs); min_clip = np.min(clipped_obs)
            #        logger.debug(f"Step obs for {agent_id} [Step {self.step_count}] clipped. Orig Max/Min: {max_orig:.2f}/{min_orig:.2f}, Clipped Max/Min: {max_clip:.2f}/{min_clip:.2f}")

            #  processed_observations[agent_id] = clipped_obs.astype(np.float32)
             processed_observations[agent_id] = obs.astype(np.float32)
        # --- Process Dones (Keep robust logic) ---
        final_terminations = {aid: terminations.get(aid, False) for aid in self._agent_ids}
        final_truncations = {aid: truncations.get(aid, False) for aid in self._agent_ids}
        final_terminations["__all__"] = all(final_terminations.values())
        final_truncations["__all__"] = all(final_truncations.values())
        if final_terminations["__all__"] or final_truncations["__all__"]:
             logger.debug(f"Wrapper: Episode ended [Internal Step {self.step_count}]. Term: {final_terminations['__all__']}, Trunc: {final_truncations['__all__']}")

        logger.debug(f"Wrapper: Returning processed obs_keys={list(processed_observations.keys())}, reward_keys={list(processed_rewards.keys())}, term_keys={list(final_terminations.keys())}, trunc_keys={list(final_truncations.keys())}")
        return processed_observations, processed_rewards, final_terminations, final_truncations, infos
    def render(self):
        return self.env.render()

    def close(self):
        logger.info("Closing RllibSatelliteEnv wrapper and underlying env.")
        self.env.close()

    @property
    def possible_agents(self):
        return self.env.possible_agents