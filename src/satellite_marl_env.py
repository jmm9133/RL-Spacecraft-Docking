# src/satellite_marl_env.py

import gymnasium as gym # Use Gymnasium instead of gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import platform
import time
import logging

# Use PettingZoo API
from pettingzoo import ParallelEnv

# Import configuration (ensure config.py is updated for PBRS and random init)
from . import config as env_config

# Get a logger for this module
logger = logging.getLogger(__name__)
# Set default level - Configure root logger in your main training script
# logging.basicConfig(level=logging.INFO) # Example basic config if run standalone

# Conditional rendering setup
try:
    import mediapy as media
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False

# --- PettingZoo Environment Factory Functions ---
def env(**kwargs):
    """
    Standard PettingZoo factory function (not typically used with RLlib directly).
    """
    env = raw_env(**kwargs)
    # from pettingzoo.utils import parallel_to_aec # Optional: Wrap for AEC API if needed
    # env = parallel_to_aec(env)
    return env

def raw_env(render_mode=None, **kwargs):
    """
    Instantiates the raw Parallel PettingZoo environment.
    This is the function used by the RLlib wrapper creator.
    """
    env = SatelliteMARLEnv(render_mode=render_mode, **kwargs)
    return env
# ---------------------------------------------

class SatelliteMARLEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for Multi-Agent Satellite Docking using MuJoCo.

    Features:
    - Potential-Based Reward Shaping (PBRS) for guiding agents.
    - Random initialization of agent states (position, velocity, orientation).
    - Support for collaborative docking (shared objective).
    - Structured for future extension to competitive mode (evasion).
    - Uses MuJoCo physics simulation.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "satellite_docking_marl_v1", # Updated name
        "render_fps": env_config.RENDER_FPS,
        "is_parallelizable": True
    }

    def __init__(self, render_mode=None, **kwargs): # Accept kwargs
        super().__init__()

        xml_path = os.path.abspath(env_config.XML_FILE_PATH)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")
        try:
             self.model = mujoco.MjModel.from_xml_path(xml_path)
             self.data = mujoco.MjData(self.model)
             logger.info(f"MuJoCo model loaded successfully from {xml_path}")
        except Exception as e:
             logger.exception(f"Error loading MuJoCo model from {xml_path}: {e}")
             raise

        self.possible_agents = env_config.POSSIBLE_AGENTS[:]
        self.agent_name_mapping = {i: agent for i, agent in enumerate(self.possible_agents)}
        self.render_mode = render_mode

        # PettingZoo spaces
        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(env_config.OBS_DIM_PER_AGENT,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(env_config.ACTION_DIM_PER_AGENT,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._get_mujoco_ids() # Cache MuJoCo element IDs

        self.renderer = None
        self.render_frames = []

        # Internal state managed by reset/step
        self.agents = []
        self.steps = 0
        self.current_actions = {}
        self.prev_potential_servicer = 0.0
        self.prev_potential_target = 0.0 # Used in competitive mode
        self.prev_docking_distance = float('inf')
        self.episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}

        # Random number generator for environment-specific randomness
        self.np_random = np.random.RandomState()

        logger.info("SatelliteMARLEnv initialized.")

    def _get_mujoco_ids(self):
        """Cache MuJoCo body, site, and joint IDs."""
        try:
            self.body_ids = {
                env_config.SERVICER_AGENT_ID: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, env_config.SERVICER_AGENT_ID),
                env_config.TARGET_AGENT_ID: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, env_config.TARGET_AGENT_ID)
            }
            self.site_ids = {
                "servicer_dock": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "servicer_dock_site"),
                "target_dock": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_dock_site")
            }
            servicer_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servicer_joint")
            target_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")

            # Error checking
            if servicer_jnt_id == -1 or target_jnt_id == -1:
                 raise ValueError("Could not find 'servicer_joint' or 'target_joint' in the MuJoCo model XML.")
            if self.body_ids[env_config.SERVICER_AGENT_ID] == -1 or self.body_ids[env_config.TARGET_AGENT_ID] == -1:
                 raise ValueError("Could not find 'servicer' or 'target' body in the MuJoCo model XML.")
            if self.site_ids["servicer_dock"] == -1 or self.site_ids["target_dock"] == -1:
                 raise ValueError("Could not find 'servicer_dock_site' or 'target_dock_site' in the MuJoCo model XML.")

            # Store qpos/qvel addresses for faster access
            self.joint_qpos_adr = {
                env_config.SERVICER_AGENT_ID: self.model.jnt_qposadr[servicer_jnt_id],
                env_config.TARGET_AGENT_ID: self.model.jnt_qposadr[target_jnt_id]
            }
            self.joint_qvel_adr = {
                env_config.SERVICER_AGENT_ID: self.model.jnt_dofadr[servicer_jnt_id],
                env_config.TARGET_AGENT_ID: self.model.jnt_dofadr[target_jnt_id]
            }
            logger.debug("MuJoCo IDs retrieved successfully.")
        except ValueError as e:
             logger.error(f"Error getting MuJoCo IDs: {e}. Check XML names.")
             raise
        except Exception as e:
             logger.exception(f"Unexpected error getting MuJoCo IDs: {e}")
             raise

    # --- PettingZoo API Methods ---
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # --- Core Environment Logic ---

    def reset(self, seed=None, options=None):
        """Resets the environment to a new random initial state."""
        logger.debug("--- Environment Reset Called (Random Initialization) ---")
        if seed is not None:
             # Seed the environment-specific random number generator
             self.np_random.seed(seed)
             logger.debug(f"Resetting with seed: {seed}")

        # Reset PettingZoo agent list and episode tracking
        self.agents = self.possible_agents[:]
        self.episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}
        self.steps = 0
        self.render_frames = []
        self.current_actions = {}

        # Reset MuJoCo simulation state
        mujoco.mj_resetData(self.model, self.data)
        logger.debug("MuJoCo data reset.")

        # --- Set Initial Conditions (Randomized) ---
        qpos_serv_start = self.joint_qpos_adr[env_config.SERVICER_AGENT_ID]
        qvel_serv_start = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
        qpos_targ_start = self.joint_qpos_adr[env_config.TARGET_AGENT_ID]
        qvel_targ_start = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]

        # Servicer Initial State
        serv_pos = self.np_random.uniform(low=env_config.INITIAL_POS_RANGE_Servicer[0], high=env_config.INITIAL_POS_RANGE_Servicer[1])
        serv_vel = self.np_random.uniform(low=env_config.INITIAL_VEL_RANGE[0], high=env_config.INITIAL_VEL_RANGE[1])
        serv_ang_vel = self.np_random.uniform(low=env_config.INITIAL_ANG_VEL_RANGE[0], high=env_config.INITIAL_ANG_VEL_RANGE[1])
        serv_quat = self.np_random.randn(4)
        serv_quat /= np.linalg.norm(serv_quat) # Normalize quaternion

        self.data.qpos[qpos_serv_start:qpos_serv_start+3] = serv_pos
        self.data.qpos[qpos_serv_start+3:qpos_serv_start+7] = serv_quat
        self.data.qvel[qvel_serv_start:qvel_serv_start+3] = serv_vel
        self.data.qvel[qvel_serv_start+3:qvel_serv_start+6] = serv_ang_vel

        # Target Initial State (Relative Offset + Random Vel/AngVel)
        targ_pos = np.zeros(3) # Initialize
        initial_dist = 0.0
        attempts = 0
        min_safe_distance = 0.8 # Prevent starting too close
        max_attempts = 20

        while initial_dist < min_safe_distance and attempts < max_attempts:
            relative_offset = self.np_random.uniform(low=env_config.INITIAL_POS_RANGE_TARGET[0], high=env_config.INITIAL_POS_RANGE_TARGET[1])
            targ_pos = serv_pos + relative_offset # Target world position
            initial_dist_vec = targ_pos - serv_pos
            initial_dist = np.linalg.norm(initial_dist_vec)
            attempts += 1
            if initial_dist < min_safe_distance:
                logger.debug(f"Reset Attempt {attempts}: Initial distance ({initial_dist:.2f}m) < {min_safe_distance}m. Resampling.")

        if initial_dist < min_safe_distance:
             logger.warning(f"Failed to find non-overlapping start after {max_attempts} attempts (dist={initial_dist:.2f}m). Placing target at default offset.")
             targ_pos = serv_pos + env_config.INITIAL_POS_OFFSET_TARGET # Fallback

        targ_vel = self.np_random.uniform(low=env_config.INITIAL_VEL_RANGE[0], high=env_config.INITIAL_VEL_RANGE[1])
        targ_ang_vel = self.np_random.uniform(low=env_config.INITIAL_ANG_VEL_RANGE[0], high=env_config.INITIAL_ANG_VEL_RANGE[1])
        targ_quat = self.np_random.randn(4)
        targ_quat /= np.linalg.norm(targ_quat) # Normalize

        self.data.qpos[qpos_targ_start:qpos_targ_start+3] = targ_pos
        self.data.qpos[qpos_targ_start+3:qpos_targ_start+7] = targ_quat
        self.data.qvel[qvel_targ_start:qvel_targ_start+3] = targ_vel
        self.data.qvel[qvel_targ_start+3:qvel_targ_start+6] = targ_ang_vel

        # Compute initial kinematics and metrics AFTER setting random state
        try:
            mujoco.mj_forward(self.model, self.data)
            logger.debug("Initial mj_forward() completed after randomization.")
        except mujoco.FatalError as e:
             logger.exception(f"MUJOCO FATAL ERROR during initial mj_forward after reset: {e}. State might be invalid.")
             # Handle this - perhaps reset again or raise an error? For now, log and continue carefully.

        # --- Initialize Previous State for PBRS ---
        dist, rel_vel_mag, orient_err = self._get_current_state_metrics()

        self.prev_potential_servicer = self._calculate_potential(dist, rel_vel_mag, orient_err)
        # Initialize target potential (even if not used in collaborative)
        # Note: The target potential function might differ in competitive mode
        self.prev_potential_target = self._calculate_potential(dist, rel_vel_mag, orient_err) # Placeholder uses same func for now

        # Use a large finite number if initial distance is invalid
        self.prev_docking_distance = dist if np.isfinite(dist) else env_config.OUT_OF_BOUNDS_DISTANCE * 2

        logger.debug(f"Reset: Initial State Metrics: Dist={dist:.4f}, RelVel={rel_vel_mag:.4f}, OrientErr={orient_err:.4f}")
        logger.debug(f"Reset: Initialized prev_potential_servicer = {self.prev_potential_servicer:.4f}")

        # Get initial observations
        observations = {}
        for agent in self.possible_agents:
            obs = self._get_obs(agent)
            # CRITICAL: Check for NaN/Inf in initial observations
            if not np.all(np.isfinite(obs)):
                 logger.error(f"!!! NaN/Inf DETECTED IN INITIAL OBSERVATION for {agent} after reset: {obs}. Clamping to zero!")
                 observations[agent] = np.zeros_like(obs, dtype=np.float32) # Return zero obs
            else:
                 observations[agent] = obs

        # Standard PettingZoo reset return format
        infos = {agent: {} for agent in self.possible_agents}

        if self.render_mode == "human":
            self.render()

        logger.debug(f"--- Environment Reset Finished (Random Init). Active agents: {self.agents} ---")
        return observations, infos

    def step(self, actions):
        """Advances the environment by one timestep."""
        step_start_time = time.time()
        logger.debug(f"--- Step {self.steps} Called (Active Agents: {self.agents}) ---")

        # Store actions for reward calculation and potential logging
        self.current_actions = actions.copy()

        # Filter actions for currently active agents, apply defaults if missing
        active_actions = {}
        for agent in self.agents:
            if agent in actions:
                active_actions[agent] = actions[agent]
            else:
                logger.warning(f"Step {self.steps}: Action missing for active agent '{agent}'. Applying default zero action.")
                active_actions[agent] = np.zeros(self.action_space(agent).shape, dtype=np.float32)

        # Apply actions to MuJoCo simulation
        try:
            self._apply_actions(active_actions)
        except Exception as e:
            logger.exception(f"CRITICAL ERROR applying actions at step {self.steps}: {e}")
            # Terminate episode immediately if action application fails
            rewards = {agent: env_config.REWARD_COLLISION for agent in self.possible_agents} # Assign penalty
            terminations = {agent: True for agent in self.possible_agents}
            truncations = {agent: False for agent in self.possible_agents}
            infos = {agent: {'status': 'action_apply_error', 'error': str(e)} for agent in self.possible_agents}
            observations = {agent: self._get_obs(agent) for agent in self.possible_agents} # Get last valid obs if possible
            self.agents = [] # End episode
            logger.error(f"Step {self.steps}: Terminating episode due to action application error.")
            # Add final episode stats to info
            self._add_final_episode_stats(infos, self.possible_agents[:]) # Use full list as all terminated
            return observations, rewards, terminations, truncations, infos

        # Step the MuJoCo simulation
        try:
             mujoco.mj_step(self.model, self.data)
             self.steps += 1
             # Post-step state logging and checks
             if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                 logger.error(f"!!! NaN/Inf detected in MuJoCo qpos/qvel immediately after mj_step {self.steps} !!!")
                 # Terminate episode if simulation becomes unstable
                 rewards = {agent: env_config.REWARD_COLLISION * 2 for agent in self.possible_agents} # Larger penalty
                 terminations = {agent: True for agent in self.possible_agents}
                 truncations = {agent: False for agent in self.possible_agents}
                 infos = {agent: {'status': 'mujoco_unstable'} for agent in self.possible_agents}
                 observations = {agent: self._get_obs(agent) for agent in self.possible_agents} # Try get last obs
                 self.agents = [] # End episode
                 logger.error(f"Step {self.steps}: Terminating episode due to MuJoCo instability.")
                 self._add_final_episode_stats(infos, self.possible_agents[:])
                 return observations, rewards, terminations, truncations, infos

        except mujoco.FatalError as e:
             logger.exception(f"MUJOCO FATAL ERROR during mj_step at step {self.steps}: {e}. Simulation unstable?")
             # Terminate episode on MuJoCo fatal error
             rewards = {agent: env_config.REWARD_COLLISION * 2 for agent in self.possible_agents} # Larger penalty
             terminations = {agent: True for agent in self.possible_agents}
             truncations = {agent: False for agent in self.possible_agents}
             infos = {agent: {'status': 'mujoco_fatal_error', 'error': str(e)} for agent in self.possible_agents}
             observations = {agent: self._get_obs(agent) for agent in self.possible_agents} # Try get last obs
             self.agents = [] # End episode
             logger.error(f"Step {self.steps}: Terminating episode due to MuJoCo fatal error.")
             self._add_final_episode_stats(infos, self.possible_agents[:])
             return observations, rewards, terminations, truncations, infos
        except Exception as e:
             logger.exception(f"Unexpected error during MuJoCo mj_step at step {self.steps}: {e}")
             # Terminate cautiously on unexpected errors too
             rewards = {agent: env_config.REWARD_COLLISION for agent in self.possible_agents}
             terminations = {agent: True for agent in self.possible_agents}
             truncations = {agent: False for agent in self.possible_agents}
             infos = {agent: {'status': 'mj_step_error', 'error': str(e)} for agent in self.possible_agents}
             observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
             self.agents = [] # End episode
             logger.error(f"Step {self.steps}: Terminating episode due to unexpected mj_step error.")
             self._add_final_episode_stats(infos, self.possible_agents[:])
             return observations, rewards, terminations, truncations, infos

        # Calculate rewards, terminations, truncations based on the new state
        rewards, terminations, truncations, infos = self._calculate_rewards_and_done()

        # Accumulate rewards and lengths for active agents
        for agent in self.agents:
            if agent in rewards:
                self.episode_rewards[agent] += rewards.get(agent, 0.0) # Use .get for safety
                self.episode_lengths[agent] += 1
            else:
                 logger.warning(f"Step {self.steps}: Agent {agent} not found in rewards dict during accumulation.")


        # Update the list of active agents based on terminations/truncations
        previous_agents = self.agents[:] # Copy before modification
        self.agents = [agent for agent in self.agents if not (terminations.get(agent, False) or truncations.get(agent, False))]
        if len(self.agents) < len(previous_agents):
            terminated_truncated_agents = set(previous_agents) - set(self.agents)
            logger.debug(f"Step {self.steps}: Agents terminated/truncated: {terminated_truncated_agents}. Remaining: {self.agents}")

        # Get observations for the *next* state for all possible agents
        # Crucial for RL algorithms needing the next observation
        observations = {}
        for agent in self.possible_agents:
             obs = self._get_obs(agent)
             # Check for NaN/Inf in observations *before* returning
             if not np.all(np.isfinite(obs)):
                  logger.error(f"!!! NaN/Inf DETECTED IN STEP OBSERVATION for {agent} at step {self.steps}. Returning zero obs!")
                  observations[agent] = np.zeros_like(obs, dtype=np.float32) # Return zero obs
             else:
                  observations[agent] = obs

        # Add final episode stats to info if episode ended for anyone THIS step
        episode_ended_this_step = len(self.agents) < len(previous_agents)
        if episode_ended_this_step:
            finished_agents = set(previous_agents) - set(self.agents)
            self._add_final_episode_stats(infos, finished_agents)

        # Rendering
        if self.render_mode == "human": self.render()
        elif self.render_mode == "rgb_array": self._render_frame()

        step_duration = time.time() - step_start_time
        logger.debug(f"--- Step {self.steps-1} Finished. Duration: {step_duration:.4f}s ---")

        # Check if all agents are done - episode truly over
        if not self.agents:
             final_statuses = {a: infos.get(a, {}).get('status', 'unknown') for a in previous_agents}
             logger.info(f"Episode ended at step {self.steps-1}. Final Statuses: {final_statuses}")

        # Return tuple: (observations, rewards, terminations, truncations, infos)
        return observations, rewards, terminations, truncations, infos

    # --- Helper Methods ---

    def _add_final_episode_stats(self, infos, agents_finished):
        """Adds final episode reward and length to the info dict for finished agents."""
        for agent_id in agents_finished:
            if agent_id in self.episode_rewards: # Check if agent exists
                # Ensure the 'episode' sub-dict exists
                if agent_id not in infos: infos[agent_id] = {}
                if 'episode' not in infos[agent_id]: infos[agent_id]['episode'] = {}

                # Add stats, making sure keys exist
                final_reward = self.episode_rewards.get(agent_id, 0.0)
                final_length = self.episode_lengths.get(agent_id, 0)
                final_status = infos.get(agent_id, {}).get('status', 'N/A')

                infos[agent_id]['episode']['r'] = final_reward
                infos[agent_id]['episode']['l'] = final_length
                infos[agent_id]['episode']['status'] = final_status # Add status for context

                logger.debug(f"Recording final episode stats for {agent_id}: R={final_reward:.2f}, L={final_length}, Status={final_status}")
            else:
                logger.warning(f"Attempted to record final stats for {agent_id}, but not found in episode tracking.")


    def _get_current_state_metrics(self):
        """Helper to get distance, relative velocity mag, and orientation error."""
        dist = float('inf')
        rel_vel_mag = float('inf')
        orient_err = np.pi # Max error default

        try:
            # Distance between docking sites
            p_s = self.data.site_xpos[self.site_ids["servicer_dock"]]
            p_t = self.data.site_xpos[self.site_ids["target_dock"]]
            if np.all(np.isfinite(p_s)) and np.all(np.isfinite(p_t)):
                dist = float(np.linalg.norm(p_t - p_s))
            else: logger.warning(f"Step {self.steps}: NaN/Inf in site positions for distance calc. Serv={p_s}, Targ={p_t}")

            # Relative Linear Velocity (using joint velocities of free joints)
            qv_s_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            qv_t_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]
            v_s = self.data.qvel[qv_s_adr:qv_s_adr+3] # Linear velocity part
            v_t = self.data.qvel[qv_t_adr:qv_t_adr+3] # Linear velocity part
            if np.all(np.isfinite(v_s)) and np.all(np.isfinite(v_t)):
                rel_vel_mag = float(np.linalg.norm(v_s - v_t))
            else: logger.warning(f"Step {self.steps}: NaN/Inf in velocities for rel_vel calc. Serv={v_s}, Targ={v_t}")

            # Orientation Error
            orient_err = self._calculate_orientation_error() # Handles internal errors

        except IndexError:
             logger.error(f"IndexError getting state metrics at step {self.steps}. MuJoCo IDs might be wrong.")
        except Exception as e:
            logger.exception(f"Error getting state metrics at step {self.steps}: {e}")

        # Clamp potentially huge values if calculation failed badly but didn't raise error
        dist = np.nan_to_num(dist, nan=env_config.OUT_OF_BOUNDS_DISTANCE*2, posinf=env_config.OUT_OF_BOUNDS_DISTANCE*2)
        rel_vel_mag = np.nan_to_num(rel_vel_mag, nan=10.0, posinf=10.0) # Clamp high speed
        orient_err = np.nan_to_num(orient_err, nan=np.pi, posinf=np.pi)

        return dist, rel_vel_mag, orient_err


    def _calculate_orientation_error(self):
        """Calculates the angular error (in radians) between docking ports' target axes."""
        try:
            # Get world rotation matrices (3x3) for the docking sites
            serv_mat = self.data.site_xmat[self.site_ids["servicer_dock"]].reshape(3, 3)
            targ_mat = self.data.site_xmat[self.site_ids["target_dock"]].reshape(3, 3)

            # Servicer port aims along its local +Z axis (index 2) in world frame
            serv_z_axis_world = serv_mat[:, 2]
            # Target port alignment axis is its local -Z axis (index 2) in world frame
            target_alignment_axis_world = -targ_mat[:, 2]

            # Ensure axes are valid vectors
            if not np.all(np.isfinite(serv_z_axis_world)) or not np.all(np.isfinite(target_alignment_axis_world)):
                 logger.warning("NaN/Inf detected in site orientation matrices.")
                 return np.pi

            norm_serv = np.linalg.norm(serv_z_axis_world)
            norm_targ = np.linalg.norm(target_alignment_axis_world)
            if norm_serv < 1e-6 or norm_targ < 1e-6:
                 logger.warning("Near-zero norm for site orientation axis.")
                 return np.pi # Avoid division by zero

            # Calculate the angle between these two vectors using dot product
            # Normalize vectors before dot product for numerical stability
            dot_prod = np.dot(serv_z_axis_world / norm_serv, target_alignment_axis_world / norm_targ)
            dot_prod = np.clip(dot_prod, -1.0, 1.0) # Clamp for arccos domain
            angle_rad = np.arccos(dot_prod)

            # Ensure angle is finite, return max error (pi) otherwise
            if not np.isfinite(angle_rad):
                logger.warning("Non-finite angle calculated in orientation error after arccos.")
                return np.pi
            return angle_rad

        except IndexError:
             logger.error(f"IndexError calculating orientation error at step {self.steps}. Site IDs might be wrong.")
             return np.pi
        except Exception as e:
            logger.exception(f"Exception calculating orientation error at step {self.steps}: {e}")
            return np.pi # Return max error on failure


    def _calculate_potential(self, distance, rel_vel_mag, orientation_error):
        """Calculates the potential function Φ based on current state. Higher is better."""
        # Get weights and epsilon from config
        Wd = env_config.POTENTIAL_WEIGHT_DISTANCE
        Wv = env_config.POTENTIAL_WEIGHT_VELOCITY
        Wo = env_config.POTENTIAL_WEIGHT_ORIENT
        EPS = env_config.POTENTIAL_DISTANCE_EPSILON

        # Use safe, clamped values for calculation
        safe_dist = max(0, distance) # Distance shouldn't be negative
        safe_vel = max(0, rel_vel_mag)
        safe_orient = max(0, min(np.pi, orientation_error)) # Clamp orientation error 0..pi

        # Potential Φ: Higher is better
        # Term 1: Increases sharply as distance -> 0
        potential_dist = Wd / (safe_dist + EPS)
        # Term 2: Penalty for relative velocity (subtract Wv * vel)
        potential_vel = -Wv * safe_vel
        # Term 3: Penalty for orientation error (subtract Wo * error)
        potential_orient = -Wo * safe_orient

        potential = potential_dist + potential_vel + potential_orient

        # Final check for non-finite potential
        if not np.isfinite(potential):
            logger.warning(f"Non-finite potential Φ={potential} calculated (Dist={distance}, Vel={rel_vel_mag}, Orient={orientation_error}). Components: D={potential_dist}, V={potential_vel}, O={potential_orient}. Clamping to 0.")
            return 0.0 # Return neutral potential on error
        return potential


    def _calculate_rewards_and_done(self):
        """
        Calculates rewards using Potential-Based Shaping (Positive Potential),
        handles terminations/truncations, and includes action costs.
        """
        rewards = {a: 0.0 for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents} # Store status like 'docked', 'collision'

        # --- 1. Get Current State Metrics ---
        dist, rel_vel_mag, orient_err = self._get_current_state_metrics()
        logger.debug(f"Step {self.steps} Rewards State: Dist={dist:.4f}, RelVel={rel_vel_mag:.4f}, OrientErr={orient_err:.4f}")

        # --- 2. Check Terminal Conditions (Docking, Collision, OOB) ---
        # Docking Check (using thresholds from config)
        docked = (dist < env_config.DOCKING_DISTANCE_THRESHOLD and
                  rel_vel_mag < env_config.DOCKING_VELOCITY_THRESHOLD and
                  orient_err < env_config.DOCKING_ORIENT_THRESHOLD)

        # Collision Check (only if not successfully docked)
        collision = False
        if not docked:
            try:
                servicer_body_id = self.body_ids[env_config.SERVICER_AGENT_ID]
                target_body_id = self.body_ids[env_config.TARGET_AGENT_ID]
                for i in range(self.data.ncon):
                    c = self.data.contact[i]
                    body1 = self.model.geom_bodyid[c.geom1]
                    body2 = self.model.geom_bodyid[c.geom2]
                    is_serv_targ_contact = ({body1, body2} == {servicer_body_id, target_body_id})
                    if is_serv_targ_contact and c.dist < 0.001: # Penetrating or near-penetrating contact
                         collision = True
                         logger.debug(f"Step {self.steps}: Collision detected! Contact {i}, dist={c.dist:.4f}")
                         break
            except Exception as e:
                logger.error(f"Error during collision check at step {self.steps}: {e}")

        # Out Of Bounds Check
        out_of_bounds = dist > env_config.OUT_OF_BOUNDS_DISTANCE

        # Apply Terminal Rewards and Set Terminations/Status
        terminal_reward = 0.0
        status = 'in_progress' # Default status
        if docked:
            logger.info(f"Step {self.steps}: Docking Successful!")
            status = 'docked'
            terminal_reward = env_config.REWARD_DOCKING_SUCCESS
            for a in self.possible_agents: terminations[a] = True
        elif collision:
            logger.info(f"Step {self.steps}: Collision Detected!")
            status = 'collision'
            terminal_reward = env_config.REWARD_COLLISION
            for a in self.possible_agents: terminations[a] = True
        elif out_of_bounds:
            logger.info(f"Step {self.steps}: Out Of Bounds (Dist={dist:.2f}m > {env_config.OUT_OF_BOUNDS_DISTANCE}m)!")
            status = 'out_of_bounds'
            terminal_reward = env_config.REWARD_OUT_OF_BOUNDS
            for a in self.possible_agents: terminations[a] = True # Terminate if they drift too far

        # Assign terminal rewards (shared in collaborative mode for now)
        # In competitive mode, target might get negative reward for docking/positive for OOB.
        for agent in self.possible_agents:
            rewards[agent] += terminal_reward
            # Store status in info dict, will be overwritten by truncation if needed
            if status != 'in_progress':
                infos[agent]['status'] = status

        # --- 3. Check Truncation Condition (Max Steps) ---
        is_truncated = False
        if self.steps >= env_config.MAX_STEPS_PER_EPISODE:
            logger.info(f"Step {self.steps}: Max steps ({env_config.MAX_STEPS_PER_EPISODE}) reached, truncating.")
            status = 'max_steps' # Truncation status overrides terminal status if both happen
            is_truncated = True
            for agent in self.possible_agents:
                # Only truncate if not already terminated by docking/collision/OOB
                if not terminations.get(agent, False):
                    truncations[agent] = True
                    # Optionally add a small penalty for timeout if not docked
                    if not docked: rewards[agent] -= 5.0 # Small timeout penalty
                # Update status regardless of termination, as truncation is the final reason
                infos[agent]['status'] = status

        episode_over = any(terminations.values()) or any(truncations.values())

        # --- 4. Calculate Shaping Rewards (PBRS + Action Cost) - If episode not over ---
        if not episode_over:
            # --- PBRS Calculation ---
            current_potential_servicer = self._calculate_potential(dist, rel_vel_mag, orient_err)
            logger.debug(f"Step {self.steps}: Potential Φ(s')_serv = {current_potential_servicer:.4f}, Prev Φ(s)_serv = {self.prev_potential_servicer:.4f}")

            gamma = env_config.POTENTIAL_GAMMA
            shaping_reward_servicer = gamma * current_potential_servicer - self.prev_potential_servicer

            if not np.isfinite(shaping_reward_servicer):
                logger.warning(f"Step {self.steps}: Non-finite PBRS reward ({shaping_reward_servicer}). Setting to 0.")
                shaping_reward_servicer = 0.0

            rewards[env_config.SERVICER_AGENT_ID] += shaping_reward_servicer
            logger.debug(f"Step {self.steps}: Shaping Reward (Servicer) = {shaping_reward_servicer:.4f}")

            # --- Action Cost Penalty ---
            for agent_id in self.possible_agents:
                 # Use agent-specific weight if competitive later, else use default
                 action_cost_weight = env_config.REWARD_WEIGHT_ACTION_COST # Default
                 if env_config.COMPETITIVE_MODE:
                      action_cost_weight = getattr(env_config, f"REWARD_WEIGHT_ACTION_COST_{agent_id.upper()}", env_config.REWARD_WEIGHT_ACTION_COST)

                 if action_cost_weight != 0 and hasattr(self, "current_actions"):
                      action = np.asarray(self.current_actions.get(agent_id, np.zeros(env_config.ACTION_DIM_PER_AGENT)))
                      action_norm_sq = np.sum(action**2) # Use squared norm penalty
                      if np.isfinite(action_norm_sq):
                           action_cost_penalty = action_cost_weight * action_norm_sq
                           rewards[agent_id] += action_cost_penalty
                           logger.debug(f"Step {self.steps}: Action Cost ({agent_id}) = {action_cost_penalty:.6f} (Weight={action_cost_weight})")
                      else:
                           logger.warning(f"Step {self.steps}: Non-finite action norm squared for {agent_id}.")

            # === Collaborative Mode: Target gets the same SHAPING reward ===
            # Action costs are already handled individually above
            if not env_config.COMPETITIVE_MODE:
                 rewards[env_config.TARGET_AGENT_ID] += shaping_reward_servicer # Share the PBRS part
                 logger.debug(f"Step {self.steps}: Shared Shaping Reward (Target) = {shaping_reward_servicer:.4f}")

            # === Competitive Mode Placeholder ===
            else: # if env_config.COMPETITIVE_MODE:
                 # Calculate target's potential (e.g., opposite goal)
                 # Wd_tgt = env_config.POTENTIAL_WEIGHT_DISTANCE_TARGET # Example
                 # Wo_tgt = env_config.POTENTIAL_WEIGHT_ORIENT_TARGET # Example
                 # potential_target = -Wd_tgt / (dist + EPS) ... # Example: Maximize distance
                 # current_potential_target = self._calculate_potential_target(...) # Define this helper

                 # shaping_reward_target = gamma * current_potential_target - self.prev_potential_target
                 # rewards[env_config.TARGET_AGENT_ID] += shaping_reward_target

                 # # Evasion bonus if timeout (handled in truncation block) - might need adjustment
                 # # Action cost already applied above with target-specific weight
                 logger.debug(f"Step {self.steps}: Competitive mode rewards calculation needed.")
                 pass # Implement competitive logic here

        # --- 5. Update Previous State for Next Step's PBRS ---
        # Calculate potential based on the state *at the end* of this step
        current_potential_servicer_end = self._calculate_potential(dist, rel_vel_mag, orient_err)
        self.prev_potential_servicer = current_potential_servicer_end

        if env_config.COMPETITIVE_MODE:
            # Calculate and store target's potential for next step
            # current_potential_target_end = self._calculate_potential_target(...)
            # self.prev_potential_target = current_potential_target_end
            pass

        # Update distance tracker (use current valid distance)
        self.prev_docking_distance = dist if np.isfinite(dist) else self.prev_docking_distance

        # --- 6. Final Reward Check & Augment Infos ---
        for agent, reward in rewards.items():
            if not np.isfinite(reward):
                logger.error(f"!!! FINAL REWARD FOR {agent} IS NON-FINITE ({reward}) at step {self.steps} !!! Setting to -50.")
                rewards[agent] = -50.0 # Assign a large penalty
            # Add total reward to info for easy logging in RLlib callbacks/results
            if agent not in infos: infos[agent] = {} # Ensure info dict exists
            infos[agent]['reward_step_total'] = rewards[agent] # Store final step reward

        logger.debug(f"Step {self.steps} Final Rewards Returned: {rewards}")
        return rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        """Calculates the observation vector for the specified agent."""
        try:
            servicer_qpos_adr = self.joint_qpos_adr[env_config.SERVICER_AGENT_ID]
            servicer_qvel_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            target_qpos_adr = self.joint_qpos_adr[env_config.TARGET_AGENT_ID]
            target_qvel_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]

            # Read raw data
            servicer_pos = self.data.qpos[servicer_qpos_adr : servicer_qpos_adr+3].copy()
            servicer_quat = self.data.qpos[servicer_qpos_adr+3 : servicer_qpos_adr+7].copy()
            servicer_vel = self.data.qvel[servicer_qvel_adr : servicer_qvel_adr+3].copy()
            servicer_ang_vel = self.data.qvel[servicer_qvel_adr+3 : servicer_qvel_adr+6].copy()

            target_pos = self.data.qpos[target_qpos_adr : target_qpos_adr+3].copy()
            target_quat = self.data.qpos[target_qpos_adr+3 : target_qpos_adr+7].copy()
            target_vel = self.data.qvel[target_qvel_adr : target_qvel_adr+3].copy()
            target_ang_vel = self.data.qvel[target_qvel_adr+3 : target_qvel_adr+6].copy()

            # --- Robust NaN/Inf checks and corrections ---
            data_list = [servicer_pos, servicer_quat, servicer_vel, servicer_ang_vel,
                         target_pos, target_quat, target_vel, target_ang_vel]
            names = ["s_pos", "s_quat", "s_vel", "s_ang", "t_pos", "t_quat", "t_vel", "t_ang"]
            needs_correction = False
            for i, arr in enumerate(data_list):
                if not np.all(np.isfinite(arr)):
                    logger.warning(f"NaN/Inf detected in raw MuJoCo data '{names[i]}' for agent {agent} at step {self.steps}: {arr}. Correcting.")
                    needs_correction = True
                    if names[i].endswith("quat"):
                         # Reset invalid quaternion to default identity
                         data_list[i][:] = [1.0, 0.0, 0.0, 0.0]
                    else:
                         # Clamp positions/velocities
                         np.nan_to_num(arr, copy=False, nan=0.0, posinf=50.0, neginf=-50.0) # Large but finite clamp
            # Reassign potentially corrected arrays back
            servicer_pos, servicer_quat, servicer_vel, servicer_ang_vel, \
            target_pos, target_quat, target_vel, target_ang_vel = data_list

            # Normalize quaternions after potential correction/reset
            norm_s = np.linalg.norm(servicer_quat)
            if norm_s > 1e-6: servicer_quat /= norm_s
            else: servicer_quat[:] = [1.0, 0.0, 0.0, 0.0] # Reset if zero norm

            norm_t = np.linalg.norm(target_quat)
            if norm_t > 1e-6: target_quat /= norm_t
            else: target_quat[:] = [1.0, 0.0, 0.0, 0.0]

            if needs_correction: logger.warning("MuJoCo data corrected before obs calculation.")
            # --- End checks ---

            # Calculate relative state (World Frame)
            # Make sure calculations use the potentially corrected data
            relative_pos_world = target_pos - servicer_pos
            relative_vel_world = target_vel - servicer_vel

            # Assemble observation based on agent ID
            if agent == env_config.SERVICER_AGENT_ID:
                # Obs: Rel Pos, Rel Vel, Servicer Quat, Servicer Ang Vel
                obs_list = [ relative_pos_world, relative_vel_world,
                                       servicer_quat, servicer_ang_vel ]
            elif agent == env_config.TARGET_AGENT_ID:
                 # Obs: (-)Rel Pos, (-)Rel Vel, Target Quat, Target Ang Vel
                 # Target sees things relative to itself
                obs_list = [ -relative_pos_world, -relative_vel_world,
                                       target_quat, target_ang_vel ]
            else:
                logger.error(f"Unknown agent ID '{agent}' requested for observation.")
                return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)

            # Check components *before* concatenation
            for i, arr in enumerate(obs_list):
                if not np.all(np.isfinite(arr)):
                    logger.error(f"!!! NaN/Inf detected in obs component {i} for agent {agent} BEFORE concatenation: {arr}. Correcting to zero.")
                    obs_list[i] = np.zeros_like(arr)

            # Concatenate into a single observation vector
            obs = np.concatenate(obs_list)

            # Final shape and finite checks
            if obs.shape[0] != env_config.OBS_DIM_PER_AGENT:
                 logger.warning(f"Obs dim mismatch for {agent} at step {self.steps}: Got {obs.shape[0]}, expected {env_config.OBS_DIM_PER_AGENT}. Fixing shape.")
                 # Pad or truncate as necessary
                 diff = env_config.OBS_DIM_PER_AGENT - obs.shape[0]
                 if diff > 0: obs = np.pad(obs, (0, diff), 'constant', constant_values=0)
                 elif diff < 0: obs = obs[:env_config.OBS_DIM_PER_AGENT]

            if not np.all(np.isfinite(obs)):
                logger.error(f"!!! NaN/Inf detected in FINAL assembled observation for {agent} at step {self.steps}. Returning zero vector.")
                return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)

            return obs.astype(np.float32) # Return correct type

        except IndexError:
             logger.exception(f"IndexError calculating observation for agent {agent} at step {self.steps}. MuJoCo IDs/addresses might be wrong.")
             return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)
        except Exception as e:
            logger.exception(f"CRITICAL Error calculating observation for agent {agent} at step {self.steps}: {e}")
            # Return a zero vector of the correct shape and type in case of unexpected errors
            return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)

    def _apply_actions(self, actions):
        """Applies actions (scaled forces/torques) to the simulation using xfrc_applied."""
        # Reset forces/torques from previous step
        if hasattr(self.data, 'xfrc_applied'):
             self.data.xfrc_applied *= 0.0 # Zero out previous forces/torques
        else:
             logger.error("MuJoCo data object missing 'xfrc_applied'. Cannot apply actions.")
             raise AttributeError("MuJoCo data object missing 'xfrc_applied'") # Raise error

        for agent, action in actions.items():
            if agent not in self.agents: continue # Skip inactive agents

            action = np.asarray(action, dtype=np.float64) # Ensure numpy array

            # Check for valid agent ID and body ID
            if agent not in self.body_ids:
                logger.warning(f"Agent {agent} has no body ID defined. Cannot apply action.")
                continue
            body_id = self.body_ids[agent]
            if body_id <= 0:
                 logger.error(f"Invalid body ID {body_id} for agent {agent}.")
                 continue

            # Calculate index for xfrc_applied (0-based, excludes world body 0)
            body_row_index = body_id - 1
            if not (0 <= body_row_index < self.data.xfrc_applied.shape[0]):
                 logger.error(f"Body index {body_row_index} (from body ID {body_id}) is out of bounds for xfrc_applied (shape {self.data.xfrc_applied.shape}).")
                 continue

            # Check action shape
            if action.shape != (env_config.ACTION_DIM_PER_AGENT,):
                 logger.warning(f"Action dim mismatch for agent {agent} at step {self.steps}: Got {action.shape}, expected {(env_config.ACTION_DIM_PER_AGENT,)}. Applying zero force/torque.")
                 self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                 continue

            # Check for NaN/Inf in raw action
            if not np.all(np.isfinite(action)):
                logger.warning(f"NaN or Inf detected in raw action for agent {agent} at step {self.steps}: {action}. Applying zero force/torque.")
                self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                continue

            # Apply scaling (Handles competitive mode implicitly if config has agent-specific values)
            force_scale = getattr(env_config, f"ACTION_FORCE_SCALING_{agent.upper()}", env_config.ACTION_FORCE_SCALING)
            torque_scale = getattr(env_config, f"ACTION_TORQUE_SCALING_{agent.upper()}", env_config.ACTION_TORQUE_SCALING)

            force = action[:3] * force_scale
            torque = action[3:] * torque_scale
            force_torque_6d = np.concatenate([force, torque])

            # Check for NaN/Inf in scaled action
            if not np.all(np.isfinite(force_torque_6d)):
                 logger.error(f"NaN or Inf detected in *SCALED* action for agent {agent} at step {self.steps}. RawAction={action}, ForceScale={force_scale}, TorqueScale={torque_scale}, Scaled={force_torque_6d}. Applying zero force/torque.")
                 self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                 continue

            # Apply the calculated force and torque to MuJoCo data
            try:
                #logger.debug(f"Applying force/torque to {agent} (body {body_id}, idx {body_row_index}): F={force}, T={torque}")
                self.data.xfrc_applied[body_row_index, :] = force_torque_6d
            except IndexError:
                 logger.error(f"IndexError assigning xfrc_applied for agent {agent} at index {body_row_index}.")
                 self.data.xfrc_applied[body_row_index, :] = np.zeros(6) # Safety zero
            except Exception as e:
                 logger.exception(f"Unexpected error assigning xfrc_applied for agent {agent} at index {body_row_index}: {e}")
                 self.data.xfrc_applied[body_row_index, :] = np.zeros(6) # Safety zero


    def render(self):
        """Renders the environment based on the render_mode."""
        if self.render_mode is None:
            # Should gym.logger.warn be used instead? PettingZoo doesn't prescribe logger yet.
            logger.warning("Render() called with render_mode=None. Set render_mode='human' or 'rgb_array' during init.")
            return None
        elif self.render_mode == "human":
            if HAS_MEDIAPY:
                 if self.renderer is None:
                      try:
                           logger.debug("Initializing MuJoCo Renderer for human mode.")
                           self.renderer = mujoco.Renderer(self.model, height=env_config.RENDER_HEIGHT, width=env_config.RENDER_WIDTH)
                           logger.info("MuJoCo Renderer initialized for human mode.")
                      except Exception as e:
                           logger.error(f"Render init error (human): {e}. Cannot render.")
                           self.render_mode = None # Disable rendering if init fails
                           return None
                 try:
                      # Ensure data state is consistent before rendering
                      # mj_forward might be redundant if just called in step, but safer
                      mujoco.mj_forward(self.model, self.data)
                      self.renderer.update_scene(self.data, camera="fixed_side") # Or use track_servicer
                      pixels = self.renderer.render()
                      if pixels is not None:
                           media.show_image(pixels)
                           # Add a small delay for visualization frame rate
                           time.sleep(1.0 / self.metadata["render_fps"])
                           return None # Human mode should return None per Gymnasium standard
                      else:
                           logger.warning("MuJoCo renderer returned None (human).")
                           return None
                 except Exception as e:
                      logger.error(f"Human rendering error: {e}")
                      # self.render_mode = None # Optionally disable on error
                      return None
            else:
                 # Warn only once about missing mediapy
                 if not hasattr(self, '_human_render_warned'):
                      logger.warning("Human rendering requires 'mediapy'. Install with 'pip install mediapy'. Rendering disabled.")
                      self._human_render_warned = True
                      self.render_mode = None # Disable if mediapy missing
                 return None
        elif self.render_mode == "rgb_array":
            frame = self._render_frame()
            return frame # Return the numpy array
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _render_frame(self):
        """Helper to render a single frame as an RGB array."""
        if self.renderer is None:
            try:
                logger.debug("Initializing MuJoCo Renderer for rgb_array mode.")
                self.renderer = mujoco.Renderer(self.model, height=env_config.RENDER_HEIGHT, width=env_config.RENDER_WIDTH)
                logger.info("MuJoCo Renderer initialized for rgb_array.")
            except Exception as e:
                 logger.error(f"Error initializing MuJoCo Renderer for rgb_array: {e}. Returning black frame.")
                 self.render_mode = None # Disable rendering if init fails
                 return np.zeros((env_config.RENDER_HEIGHT, env_config.RENDER_WIDTH, 3), dtype=np.uint8)
        try:
            # Ensure data state is consistent
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera="fixed_side") # Or other camera
            pixels = self.renderer.render()
            if pixels is not None:
                # logger.debug(f"Rendered frame for rgb_array")
                return pixels.astype(np.uint8)
            else:
                logger.warning("MuJoCo renderer returned None for rgb_array frame. Returning black frame.")
                return np.zeros((env_config.RENDER_HEIGHT, env_config.RENDER_WIDTH, 3), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error during frame rendering for rgb_array: {e}. Returning black frame.")
            # Consider disabling rendering if errors persist
            # self.render_mode = None
            return np.zeros((env_config.RENDER_HEIGHT, env_config.RENDER_WIDTH, 3), dtype=np.uint8)


    def close(self):
        """Closes the environment and releases resources."""
        if self.renderer:
            try:
                logger.debug("Closing MuJoCo renderer.")
                self.renderer.close()
            except Exception as e:
                logger.error(f"Error closing MuJoCo renderer: {e}")
            self.renderer = None
        # No need to explicitly delete model/data unless managing memory strictly
        logger.info("SatelliteMARLEnv closed.")