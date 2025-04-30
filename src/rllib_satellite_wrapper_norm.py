import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from . import config as env_config # Ensure config is imported
from .satellite_marl_env import raw_env as satellite_pettingzoo_creator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    clipping for robustness and reward normalization for training stability.
    """
    def __init__(self, env_config_dict=None):
    # Initialize with empty dict if None
        if env_config_dict is None: 
            env_config_dict = {}
        
        # Get render mode and other existing parameters
        render_mode = env_config_dict.get("render_mode", None)
        self.env = satellite_pettingzoo_creator(render_mode=render_mode, **env_config_dict)
        self._agent_ids = set(self.env.possible_agents)
        self.observation_space = gym.spaces.Dict({
            agent: self.env.observation_space(agent) for agent in self.env.possible_agents
        })
        self.action_space = gym.spaces.Dict({
            agent: self.env.action_space(agent) for agent in self.env.possible_agents
        })
        
        # --- Reward Normalization Parameters ---
        # Defaults adjusted for large terminal rewards 
        self.normalize_rewards = env_config_dict.get("normalize_rewards", True)
        
        # Use a smaller scaling factor to handle the large range of rewards
        self.reward_scaling = env_config_dict.get("reward_scaling", 0.01)
        
        # Enable post-normalization clipping
        self.reward_clip_after_norm = env_config_dict.get("reward_clip_after_norm", True)
        
        # Wider clip range to accommodate both small shaping and large terminal rewards
        self.reward_norm_clip_min = env_config_dict.get("reward_norm_clip_min", -200.0)
        self.reward_norm_clip_max = env_config_dict.get("reward_norm_clip_max", 200.0)
        
        # Optionally: Add a parameter for reward history window size
        self.reward_history_window = env_config_dict.get("reward_history_window", 10000)
        
        # Initialize running statistics for each agent
        self.reward_running_mean = {}
        self.reward_running_var = {}
        self.reward_running_count = {}
        
        # Optional: Keep a history of recent rewards for better statistics
        self.reward_history = {}
        
        for agent_id in self._agent_ids:
            self.reward_running_mean[agent_id] = 0.0
            self.reward_running_var[agent_id] = 1.0
            self.reward_running_count[agent_id] = 0
            self.reward_history[agent_id] = []
        
        self.step_count = 0
        super().__init__()
        
        logger.info(f"RllibSatelliteEnv wrapper initialized for agents: {self._agent_ids}")
        logger.info(f"Reward normalization {'enabled' if self.normalize_rewards else 'disabled'}, " 
                    f"scaling={self.reward_scaling}, " 
                    f"post-norm clipping={'enabled' if self.reward_clip_after_norm else 'disabled'} " 
                    f"to [{self.reward_norm_clip_min}, {self.reward_norm_clip_max}]")

    def reset(self, *, seed=None, options=None):
        self.episode_raw_rewards = {agent_id: 0.0 for agent_id in self._agent_ids}
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

        # Reset step counter on environment reset
        self.step_count = 0
        logger.debug(f"Wrapper: Processed reset observations keys: {list(processed_observations.keys())}")
        return processed_observations, infos


    def step(self, action_dict):
        if not hasattr(self, 'step_count'): self.step_count = 0
        self.step_count += 1
        log_frequency = 200  # Log rewards less frequently once stable

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

        # --- Process Rewards (with normalization) ---
        processed_rewards = {}
        raw_rewards_log = {}
        normalized_rewards_log = {}
        any_bad_raw_reward = False
        
        for agent_id in self._agent_ids:
            raw_reward = rewards.get(agent_id)
            self.episode_raw_rewards[agent_id] += raw_reward
            raw_rewards_log[agent_id] = raw_reward
            
            # Handle None or non-finite rewards
            if raw_reward is None:
                processed_reward = 0.0
                any_bad_raw_reward = True
                # Add to processed_rewards dictionary
                processed_rewards[agent_id] = processed_reward
            elif not np.isfinite(raw_reward):
                logger.error(f"!!! Non-finite reward {raw_reward} from raw env for {agent_id} [Step {self.step_count}]! Clipping to zero. !!!")
                processed_reward = 0.0
                any_bad_raw_reward = True
                # Add to processed_rewards dictionary
                processed_rewards[agent_id] = processed_reward
            else:
                # First clip the raw reward to reasonable bounds
                clipped_reward = np.clip(float(raw_reward), REWARD_CLIP_MIN, REWARD_CLIP_MAX)
                
                # Then normalize if enabled
                if self.normalize_rewards:
                    # Keep a history of recent rewards to compute better statistics
                    self.reward_history[agent_id].append(clipped_reward)
                    if len(self.reward_history[agent_id]) > self.reward_history_window:
                        self.reward_history[agent_id].pop(0)
                    
                    # Update running statistics with Welford's algorithm
                    self.reward_running_count[agent_id] += 1
                    count = self.reward_running_count[agent_id]
                    
                    delta = clipped_reward - self.reward_running_mean[agent_id]
                    self.reward_running_mean[agent_id] += delta / count
                    delta2 = clipped_reward - self.reward_running_mean[agent_id]
                    self.reward_running_var[agent_id] += delta * delta2
                    
                    # Compute std dev with a small epsilon to avoid division by zero
                    # Only start using std after collecting enough samples
                    if count > 100:  # Wait for more samples given the large reward disparity
                        std = np.sqrt(self.reward_running_var[agent_id] / count + 1e-8)
                    else:
                        # Start with a large initial std to prevent over-scaling early rewards
                        std = max(1000.0, np.std(self.reward_history[agent_id]) if self.reward_history[agent_id] else 1000.0)
                    
                    # Add info about reward values for debugging
                    if agent_id not in infos:
                        infos[agent_id] = {}
                    infos[agent_id]["raw_reward"] = raw_reward
                    infos[agent_id]["clipped_reward"] = clipped_reward
                    
                    # Normalize the reward
                    normalized_reward = (clipped_reward - self.reward_running_mean[agent_id]) / std
                    
                    # Apply scaling factor
                    scaled_reward = normalized_reward * self.reward_scaling
                    
                    # Post-normalization clipping
                    if self.reward_clip_after_norm:
                        processed_reward = np.clip(scaled_reward, 
                                                self.reward_norm_clip_min, 
                                                self.reward_norm_clip_max)
                    else:
                        processed_reward = scaled_reward
                    
                    # Add more debug info
                    infos[agent_id]["normalized_reward"] = normalized_reward
                    infos[agent_id]["scaled_reward"] = scaled_reward
                    infos[agent_id]["processed_reward"] = processed_reward
                
                    normalized_rewards_log[agent_id] = processed_reward
                else:
                    # If not normalizing, just use the clipped reward
                    processed_reward = clipped_reward
                
                # Add to processed_rewards dictionary (CRITICAL FIX)
                processed_rewards[agent_id] = processed_reward

        
        # Log processed rewards
        if (self.step_count % log_frequency == 1) or any_bad_raw_reward:
            rewards_str = ", ".join([f"{k}: {v:.3f}" for k, v in processed_rewards.items()])
            raw_rewards_str = ", ".join([f"{k}: {v}" for k, v in raw_rewards_log.items()])
            
            log_msg = f"Wrapper Step {self.step_count}: Processed Rewards: {{{rewards_str}}} (Raw: {{{raw_rewards_str}}})"
            
            # Add normalization stats if enabled
            if self.normalize_rewards:
                means_str = ", ".join([f"{k}: {v:.3f}" for k, v in self.reward_running_mean.items()])
                stds_str = ", ".join([f"{k}: {v:.3f}" for k, v in self.reward_running_var.items() if self.reward_running_count[k] > 1])
                norm_rewards_str = ", ".join([f"{k}: {v:.3f}" for k, v in normalized_rewards_log.items()])
                
                log_msg += f"\nNormalization Stats - Means: {{{means_str}}}, Variances: {{{stds_str}}}, Normalized: {{{norm_rewards_str}}}"
            
            logger.info(log_msg)

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

            processed_observations[agent_id] = obs.astype(np.float32)
            
        # --- Process Dones (Keep robust logic) ---
        if terminations.get(agent_id, False) or truncations.get(agent_id, False):
            infos[agent_id]["episode_raw_reward"] = self.episode_raw_rewards[agent_id]
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