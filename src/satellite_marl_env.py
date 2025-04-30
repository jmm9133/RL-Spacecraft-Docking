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
from . import config2 as env_config

# Get a logger for this module
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Set level in main script

# Conditional rendering setup (Keep as is)
try:
    import mediapy as media
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False

# --- PettingZoo Environment Factory Functions (Keep as is) ---
def env(**kwargs):
    env = raw_env(**kwargs)
    return env

def raw_env(render_mode=None, **kwargs):
    env = SatelliteMARLEnv(render_mode=render_mode, **kwargs)
    return env
# ---------------------------------------------

class SatelliteMARLEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for Multi-Agent Satellite Docking using MuJoCo.
    (Docstring remains the same)
    """
    metadata = { # Keep as is
        "render_modes": ["human", "rgb_array"],
        "name": "satellite_docking_marl_v1",
        "render_fps": env_config.RENDER_FPS,
        "is_parallelizable": True
    }

    def __init__(self, render_mode=None, **kwargs): # Keep as is
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

        # PettingZoo spaces (Keep as is)
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

        # Internal state managed by reset/step (Keep as is)
        self.agents = []
        self.steps = 0
        self.current_actions = {}
        self.prev_potential_servicer = 0.0
        self.prev_potential_target = 0.0
        self.prev_docking_distance = float('inf')
        self.episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}

        self.np_random = np.random.RandomState() # Keep as is

        logger.info("SatelliteMARLEnv initialized.")

    def _get_mujoco_ids(self): # Keep as is
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

            if servicer_jnt_id == -1 or target_jnt_id == -1:
                 raise ValueError("Could not find 'servicer_joint' or 'target_joint' in the MuJoCo model XML.")
            if self.body_ids[env_config.SERVICER_AGENT_ID] == -1 or self.body_ids[env_config.TARGET_AGENT_ID] == -1:
                 raise ValueError("Could not find 'servicer' or 'target' body in the MuJoCo model XML.")
            if self.site_ids["servicer_dock"] == -1 or self.site_ids["target_dock"] == -1:
                 raise ValueError("Could not find 'servicer_dock_site' or 'target_dock_site' in the MuJoCo model XML.")

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

    # --- PettingZoo API Methods (Keep as is) ---
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # --- Core Environment Logic ---

    def reset(self, seed=None, options=None): # Keep random initialization, add obs checks
        """Resets the environment to a new random initial state."""
        logger.debug("--- Environment Reset Called (Random Initialization) ---")
        if seed is not None:
             self.np_random.seed(seed)
             logger.debug(f"Resetting with seed: {seed}")

        self.agents = self.possible_agents[:]
        self.episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}
        self.steps = 0
        self.render_frames = []
        self.current_actions = {}

        mujoco.mj_resetData(self.model, self.data)
        logger.debug("MuJoCo data reset.")

        # --- Set Initial Conditions (Randomized - Keep existing logic) ---
        qpos_serv_start = self.joint_qpos_adr[env_config.SERVICER_AGENT_ID]
        qvel_serv_start = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
        qpos_targ_start = self.joint_qpos_adr[env_config.TARGET_AGENT_ID]
        qvel_targ_start = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]

        serv_pos = self.np_random.uniform(low=env_config.INITIAL_POS_RANGE_Servicer[0], high=env_config.INITIAL_POS_RANGE_Servicer[1])
        serv_vel = self.np_random.uniform(low=env_config.INITIAL_VEL_RANGE[0], high=env_config.INITIAL_VEL_RANGE[1])
        serv_ang_vel = self.np_random.uniform(low=env_config.INITIAL_ANG_VEL_RANGE[0], high=env_config.INITIAL_ANG_VEL_RANGE[1])
        serv_quat = self.np_random.randn(4); serv_quat /= np.linalg.norm(serv_quat)
        serv_quat[0] = 0.0; serv_quat[1] = 0.0; serv_quat[2] = 0.0; serv_quat[3] = 1.0 # Fixed orientation

        self.data.qpos[qpos_serv_start:qpos_serv_start+3] = serv_pos
        self.data.qpos[qpos_serv_start+3:qpos_serv_start+7] = serv_quat
        self.data.qvel[qvel_serv_start:qvel_serv_start+3] = serv_vel
        self.data.qvel[qvel_serv_start+3:qvel_serv_start+6] = serv_ang_vel

        targ_pos = np.zeros(3); initial_dist = 0.0; attempts = 0
        min_safe_distance = 0.8; max_attempts = 20
        while initial_dist < min_safe_distance and attempts < max_attempts:
            relative_offset = self.np_random.uniform(low=env_config.INITIAL_POS_RANGE_TARGET[0], high=env_config.INITIAL_POS_RANGE_TARGET[1])
            targ_pos = serv_pos + relative_offset
            initial_dist_vec = targ_pos - serv_pos
            initial_dist = np.linalg.norm(initial_dist_vec)
            attempts += 1
            if initial_dist < min_safe_distance: logger.debug(f"Reset Attempt {attempts}: Initial distance ({initial_dist:.2f}m) < {min_safe_distance}m. Resampling.")
        if initial_dist < min_safe_distance:
             logger.warning(f"Failed to find non-overlapping start after {max_attempts} attempts (dist={initial_dist:.2f}m). Placing target at default offset.")
             targ_pos = serv_pos + env_config.INITIAL_POS_OFFSET_TARGET

        targ_vel = self.np_random.uniform(low=env_config.INITIAL_VEL_RANGE[0], high=env_config.INITIAL_VEL_RANGE[1])
        targ_ang_vel = self.np_random.uniform(low=env_config.INITIAL_ANG_VEL_RANGE[0], high=env_config.INITIAL_ANG_VEL_RANGE[1])
        targ_quat = self.np_random.randn(4); targ_quat /= np.linalg.norm(targ_quat)
        #targ_quat[0] = 0.0; targ_quat[1] = 0.0; targ_quat[2] = 0.0; targ_quat[3] = 1.0 # Fixed orientation
        targ_quat = [0.0, 0.7071, 0.0, 0.7071]

        self.data.qpos[qpos_targ_start:qpos_targ_start+3] = targ_pos
        self.data.qpos[qpos_targ_start+3:qpos_targ_start+7] = targ_quat
        self.data.qvel[qvel_targ_start:qvel_targ_start+3] = targ_vel
        self.data.qvel[qvel_targ_start+3:qvel_targ_start+6] = targ_ang_vel

        # --- Compute initial state and PBRS ---
        try:
            mujoco.mj_forward(self.model, self.data)
            logger.debug("Initial mj_forward() completed after randomization.")
            dist, rel_vel_mag,closing_rate, orient_err = self._get_current_state_metrics() # Get metrics AFTER forward
            self.prev_potential_servicer = self._calculate_potential(dist, rel_vel_mag,closing_rate, orient_err)
            self.prev_potential_target = self._calculate_potential(dist, rel_vel_mag,closing_rate, orient_err) # Placeholder
            self.prev_docking_distance = dist if np.isfinite(dist) else env_config.OUT_OF_BOUNDS_DISTANCE * 2
            logger.debug(f"Reset: Initial State Metrics: Dist={dist:.4f}, RelVel={rel_vel_mag:.4f}, OrientErr={orient_err:.4f}")
            logger.debug(f"Reset: Initialized prev_potential_servicer = {self.prev_potential_servicer:.4f}")
        except mujoco.FatalError as e:
             logger.exception(f"MUJOCO FATAL ERROR during initial mj_forward/metrics after reset: {e}. State might be invalid.")
             # Return zero obs and empty info if state is corrupt from start
             observations = {agent: np.zeros(self.observation_space(agent).shape, dtype=np.float32) for agent in self.possible_agents}
             infos = {agent: {'status': 'reset_mujoco_error', 'error': str(e)} for agent in self.possible_agents}
             self.agents = [] # Prevent steps
             return observations, infos
        except Exception as e:
            logger.exception(f"Unexpected ERROR during initial metrics/potential calc after reset: {e}")
            observations = {agent: np.zeros(self.observation_space(agent).shape, dtype=np.float32) for agent in self.possible_agents}
            infos = {agent: {'status': 'reset_metric_error', 'error': str(e)} for agent in self.possible_agents}
            self.agents = []
            return observations, infos


        # Get initial observations using the robust _get_obs
        observations = {}
        for agent in self.possible_agents:
            obs = self._get_obs(agent) # _get_obs now handles internal NaNs
            observations[agent] = obs # _get_obs returns zero vector on failure

        infos = {agent: {} for agent in self.possible_agents}

        if self.render_mode == "human": self.render()

        logger.debug(f"--- Environment Reset Finished. Active agents: {self.agents} ---")
        return observations, infos

    def step(self, actions):
        """Advances the environment by one timestep."""
        step_start_time = time.time()
        logger.debug(f"--- Step {self.steps} Called (Active Agents: {self.agents}) ---")

        if not self.agents: # If no agents active (e.g., due to error in reset), return immediately
             logger.warning(f"Step {self.steps} called with no active agents. Returning empty step.")
             # Return dummy values consistent with PettingZoo API for a finished episode
             observations = {agent: self._get_obs(agent) for agent in self.possible_agents} # Get last obs state
             rewards = {agent: 0.0 for agent in self.possible_agents}
             terminations = {agent: True for agent in self.possible_agents}
             truncations = {agent: False for agent in self.possible_agents}
             infos = {agent: {'status': 'no_active_agents'} for agent in self.possible_agents}
             terminations["__all__"] = True
             truncations["__all__"] = False
             return observations, rewards, terminations, truncations, infos


        self.current_actions = actions.copy()
        active_actions = {agent: actions.get(agent, np.zeros(self.action_space(agent).shape, dtype=np.float32))
                          for agent in self.agents}

        # Apply actions (robust _apply_actions handles internal errors)
        try:
            self._apply_actions(active_actions)
        except Exception as e:
            logger.exception(f"CRITICAL ERROR applying actions at step {self.steps}: {e}")
            rewards = {agent: env_config.REWARD_COLLISION for agent in self.possible_agents}
            terminations = {agent: True for agent in self.possible_agents}
            truncations = {agent: False for agent in self.possible_agents}
            infos = {agent: {'status': 'action_apply_error', 'error': str(e)} for agent in self.possible_agents}
            observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
            self.agents = []
            terminations["__all__"] = True; truncations["__all__"] = False
            self._add_final_episode_stats(infos, self.possible_agents[:])
            return observations, rewards, terminations, truncations, infos

        # Step MuJoCo simulation (robust error handling)
        try:
             mujoco.mj_step(self.model, self.data)
             self.steps += 1
             if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                 logger.error(f"!!! NaN/Inf detected in MuJoCo qpos/qvel immediately after mj_step {self.steps} !!! State: qpos={self.data.qpos} qvel={self.data.qvel}")
                 raise mujoco.FatalError("NaN/Inf in MuJoCo state after mj_step") # Force termination

        except mujoco.FatalError as e:
             logger.exception(f"MUJOCO FATAL ERROR during mj_step at step {self.steps}: {e}. Simulation unstable?")
             rewards = {agent: env_config.REWARD_COLLISION * 2 for agent in self.possible_agents} # Larger penalty
             terminations = {agent: True for agent in self.possible_agents}
             truncations = {agent: False for agent in self.possible_agents}
             infos = {agent: {'status': 'mujoco_fatal_error', 'error': str(e)} for agent in self.possible_agents}
             observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
             self.agents = []
             terminations["__all__"] = True; truncations["__all__"] = False
             self._add_final_episode_stats(infos, self.possible_agents[:])
             return observations, rewards, terminations, truncations, infos
        except Exception as e:
             logger.exception(f"Unexpected error during MuJoCo mj_step at step {self.steps}: {e}")
             rewards = {agent: env_config.REWARD_COLLISION for agent in self.possible_agents}
             terminations = {agent: True for agent in self.possible_agents}
             truncations = {agent: False for agent in self.possible_agents}
             infos = {agent: {'status': 'mj_step_error', 'error': str(e)} for agent in self.possible_agents}
             observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
             self.agents = []
             terminations["__all__"] = True; truncations["__all__"] = False
             self._add_final_episode_stats(infos, self.possible_agents[:])
             return observations, rewards, terminations, truncations, infos

        # Calculate rewards, terminations, truncations (robust methods handle internal errors)
        try:
            rewards, terminations, truncations, infos = self._calculate_rewards_and_done()
            # ——— SANITIZE ANY NaN/INF IN THE REWARDS BEFORE WE ADD THEM INTO episode_rewards ———
            for agent, r in rewards.items():
                if not np.isfinite(r):
                    logger.error(f"Step {self.steps}: Non-finite reward ({r}) for {agent}. Clamping to 0.")
                    rewards[agent] = 0.0
        except Exception as e:
            logger.exception(f"CRITICAL ERROR calculating rewards/done at step {self.steps}: {e}")
            rewards = {agent: env_config.REWARD_COLLISION for agent in self.possible_agents}
            terminations = {agent: True for agent in self.possible_agents} # Terminate on reward error
            truncations = {agent: False for agent in self.possible_agents}
            infos = {agent: {'status': 'reward_calc_error', 'error': str(e)} for agent in self.possible_agents}
            # Observations below will use the state right after the failed reward calc
            self.agents = [] # Terminate episode
            terminations["__all__"] = True; truncations["__all__"] = False


        # Accumulate rewards/lengths (safe .get)
        for agent in self.agents:
            self.episode_rewards[agent] += rewards.get(agent, 0.0)
            self.episode_lengths[agent] += 1

        # Update active agents list
        previous_agents = self.agents[:]
        finished_agents_this_step = set()
        if any(terminations.values()) or any(truncations.values()): # Check if anyone finished
             self.agents = [agent for agent in self.agents if not (terminations.get(agent, False) or truncations.get(agent, False))]
             finished_agents_this_step = set(previous_agents) - set(self.agents)
             if finished_agents_this_step:
                 logger.debug(f"Step {self.steps}: Agents finished: {finished_agents_this_step}. Remaining: {self.agents}")

        # Get observations for the *next* state for ALL agents
        observations = {}
        for agent in self.possible_agents:
             obs = self._get_obs(agent) # _get_obs is robust
             observations[agent] = obs

        # Add final episode stats to info if episode ended for anyone THIS step
        if finished_agents_this_step:
             self._add_final_episode_stats(infos, finished_agents_this_step)

        # Add __all__ keys REQUIRED by RLlib wrapper / multi-agent API
        # Calculate based on the potentially modified terminations/truncations from reward calc
        terminations["__all__"] = all(terminations.get(a, False) for a in self.possible_agents)
        truncations["__all__"] = all(truncations.get(a, False) for a in self.possible_agents)

        # Rendering (Keep as is)
        if self.render_mode == "human": self.render()
        elif self.render_mode == "rgb_array": self._render_frame()

        step_duration = time.time() - step_start_time
        logger.debug(f"--- Step {self.steps-1} Finished. Duration: {step_duration:.4f}s ---")

        # Log final status if episode ended for all
        if not self.agents and previous_agents: # Ensure it was previously active
             final_statuses = {a: infos.get(a, {}).get('status', 'unknown') for a in previous_agents}
             logger.info(f"Episode ended at step {self.steps-1}. Final Statuses: {final_statuses}")

        return observations, rewards, terminations, truncations, infos

    # --- Helper Methods ---

    def _add_final_episode_stats(self, infos, agents_finished): # Keep as is
        """Adds final episode reward and length to the info dict for finished agents."""
        for agent_id in agents_finished:
            if agent_id in self.possible_agents: # Check against possible_agents
                if agent_id not in infos: infos[agent_id] = {}
                if 'episode' not in infos[agent_id]: infos[agent_id]['episode'] = {}

                final_reward = self.episode_rewards.get(agent_id, 0.0)
                final_length = self.episode_lengths.get(agent_id, 0)
                final_status = infos.get(agent_id, {}).get('status', 'N/A')

                infos[agent_id]['episode']['r'] = final_reward
                infos[agent_id]['episode']['l'] = final_length
                infos[agent_id]['episode']['status'] = final_status

                logger.debug(f"Recording final episode stats for {agent_id}: R={final_reward:.2f}, L={final_length}, Status={final_status}")
            else:
                logger.warning(f"Attempted to record final stats for invalid agent ID {agent_id}.")


    ''' def _get_current_state_metrics(self): # Add more nan_to_num/logging
        """Helper to get distance, relative velocity mag, and orientation error."""
        dist = float('inf')
        rel_vel_mag = float('inf')
        orient_err = np.pi # Max error default

        try:
            # Distance
            p_s = self.data.site_xpos[self.site_ids["servicer_dock"]]
            p_t = self.data.site_xpos[self.site_ids["target_dock"]]
            if not np.all(np.isfinite(p_s)) or not np.all(np.isfinite(p_t)):
                logger.warning(f"Step {self.steps}: NaN/Inf in site positions for distance calc. Serv={p_s}, Targ={p_t}. Using max distance.")
                dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2 # Use large finite value
            else:
                dist = float(np.linalg.norm(p_t - p_s))
                if not np.isfinite(dist): # Check norm result itself
                     logger.warning(f"Step {self.steps}: Non-finite distance calculated ({dist}). Using max.")
                     dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2


            # Relative Linear Velocity
            qv_s_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            qv_t_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]
            v_s = self.data.qvel[qv_s_adr:qv_s_adr+3]
            v_t = self.data.qvel[qv_t_adr:qv_t_adr+3]
            if not np.all(np.isfinite(v_s)) or not np.all(np.isfinite(v_t)):
                logger.warning(f"Step {self.steps}: NaN/Inf in velocities for rel_vel calc. Serv={v_s}, Targ={v_t}. Using large rel_vel.")
                rel_vel_mag = 50.0 # Assign large finite velocity
            else:
                rel_vel_vec = v_s - v_t
                rel_vel_mag = float(np.linalg.norm(rel_vel_vec))
                if not np.isfinite(rel_vel_mag):
                     logger.warning(f"Step {self.steps}: Non-finite rel_vel_mag calculated ({rel_vel_mag}). Using large.")
                     rel_vel_mag = 50.0

            # Orientation Error (robust _calculate_orientation_error handles internal errors)
            orient_err = self._calculate_orientation_error()
            if not np.isfinite(orient_err): # Ensure it's finite after the calculation
                 logger.warning(f"Step {self.steps}: Non-finite orientation error ({orient_err}) returned. Using max (pi).")
                 orient_err = np.pi

        except IndexError:
             logger.error(f"IndexError getting state metrics at step {self.steps}. MuJoCo IDs might be wrong. Using default max values.")
             dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2
             rel_vel_mag = 50.0
             orient_err = np.pi
        except Exception as e:
            logger.exception(f"Error getting state metrics at step {self.steps}: {e}. Using default max values.")
            dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2
            rel_vel_mag = 50.0
            orient_err = np.pi

        # Final safety clamp/nan_to_num just in case
        dist = np.nan_to_num(dist, nan=env_config.OUT_OF_BOUNDS_DISTANCE*2, posinf=env_config.OUT_OF_BOUNDS_DISTANCE*2, neginf=0.0) # Dist shouldn't be negative
        rel_vel_mag = np.nan_to_num(rel_vel_mag, nan=50.0, posinf=50.0, neginf=0.0) # Vel mag shouldn't be negative
        orient_err = np.nan_to_num(orient_err, nan=np.pi, posinf=np.pi, neginf=0.0) # Orient err 0 to pi

        return dist, rel_vel_mag, orient_err'''

    def _get_current_state_metrics(self): # Add more nan_to_num/logging
        """Helper to get distance, relative velocity mag, approach rate, and orientation error."""
        dist = float('inf')
        rel_vel_mag = float('inf')
        closing_rate = 0.0  # Positive = approaching, Negative = receding
        orient_err = np.pi  # Max error default

        try:
            # --- Distance ---
            p_s = self.data.site_xpos[self.site_ids["servicer_dock"]]
            p_t = self.data.site_xpos[self.site_ids["target_dock"]]
            if not np.all(np.isfinite(p_s)) or not np.all(np.isfinite(p_t)):
                logger.warning(f"Step {self.steps}: NaN/Inf in site positions. Serv={p_s}, Targ={p_t}")
                dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2
            else:
                approach_vec = p_t - p_s
                dist = float(np.linalg.norm(approach_vec))
                if not np.isfinite(dist):
                    logger.warning(f"Step {self.steps}: Non-finite distance ({dist}).")
                    dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2

            # --- Relative Velocity ---
            qv_s_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            qv_t_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]
            v_s = self.data.qvel[qv_s_adr:qv_s_adr+3]
            v_t = self.data.qvel[qv_t_adr:qv_t_adr+3]
            if not np.all(np.isfinite(v_s)) or not np.all(np.isfinite(v_t)):
                logger.warning(f"Step {self.steps}: NaN/Inf in velocities. Serv={v_s}, Targ={v_t}")
                rel_vel_mag = 50.0
            else:
                rel_vel_vec = v_s - v_t
                rel_vel_mag = float(np.linalg.norm(rel_vel_vec))
                if not np.isfinite(rel_vel_mag):
                    logger.warning(f"Step {self.steps}: Non-finite rel_vel_mag ({rel_vel_mag}).")
                    rel_vel_mag = 50.0

                # --- Closing Rate ---
                if dist > 1e-6:
                    unit_approach_vec = approach_vec / dist
                    closing_rate = -np.dot(rel_vel_vec, unit_approach_vec)  # Positive if approaching
                else:
                    closing_rate = 0.0  # Already docked or nearly so

            # --- Orientation Error ---
            orient_err = self._calculate_orientation_error()
            if not np.isfinite(orient_err):
                logger.warning(f"Step {self.steps}: Non-finite orientation error ({orient_err}).")
                orient_err = np.pi

        except IndexError:
            logger.error(f"IndexError getting state metrics at step {self.steps}.")
            dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2
            rel_vel_mag = 50.0
            closing_rate = 0.0
            orient_err = np.pi
        except Exception as e:
            logger.exception(f"Error getting state metrics at step {self.steps}: {e}")
            dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2
            rel_vel_mag = 50.0
            closing_rate = 0.0
            orient_err = np.pi

        # Final clamping
        dist = np.nan_to_num(dist, nan=env_config.OUT_OF_BOUNDS_DISTANCE*2, posinf=env_config.OUT_OF_BOUNDS_DISTANCE*2, neginf=0.0)
        rel_vel_mag = np.nan_to_num(rel_vel_mag, nan=50.0, posinf=50.0, neginf=0.0)
        closing_rate = np.nan_to_num(closing_rate, nan=0.0, posinf=50.0, neginf=-50.0)
        orient_err = np.nan_to_num(orient_err, nan=np.pi, posinf=np.pi, neginf=0.0)

        return dist, rel_vel_mag, closing_rate, orient_err
    def _calculate_orientation_guidance(self):
        """
        Calculates orientation error and guidance vector for docking alignment.
        Returns:
            - angle_error: Scalar angle between docking ports (radians)
            - error_axis: Unit vector representing axis to rotate around for correction
        """
        # Get docking port orientations
        serv_dock_mat = self.data.site_xmat[self.site_ids["servicer_dock"]].reshape(3, 3)
        targ_dock_mat = self.data.site_xmat[self.site_ids["target_dock"]].reshape(3, 3)
        
        # Extract the docking axis (z-axis) from both ports
        serv_dock_axis = serv_dock_mat[:, 2]  # Z-axis of servicer dock
        targ_dock_axis = -targ_dock_mat[:, 2]  # Negative Z-axis of target dock (for alignment)
        
        # Normalize axes to ensure unit vectors (defensive programming)
        serv_norm = np.linalg.norm(serv_dock_axis)
        targ_norm = np.linalg.norm(targ_dock_axis)
        
        if serv_norm < 1e-6 or targ_norm < 1e-6:
            logger.warning(f"Near-zero dock axis norm detected: serv={serv_norm}, targ={targ_norm}")
            return np.pi, np.zeros(3)  # Return max error and zero guidance
        
        serv_dock_axis = serv_dock_axis / serv_norm
        targ_dock_axis = targ_dock_axis / targ_norm
        
        # Calculate the angle between the two vectors
        dot_product = np.clip(np.dot(serv_dock_axis, targ_dock_axis), -1.0, 1.0)
        angle_error = np.arccos(dot_product)
        
        # Calculate the axis to rotate around (direction of correction)
        error_axis = np.cross(serv_dock_axis, targ_dock_axis)
        axis_norm = np.linalg.norm(error_axis)
        
        # If axes are nearly aligned or anti-aligned, the cross product is near zero
        if axis_norm < 1e-6:
            if dot_product > 0:
                # Nearly aligned already
                return angle_error, np.zeros(3)
            else:
                # Anti-aligned (need 180° rotation)
                # Choose any perpendicular axis (e.g., using the UP vector as reference)
                world_up = np.array([0.0, 0.0, 1.0])
                error_axis = np.cross(serv_dock_axis, world_up)
                axis_norm = np.linalg.norm(error_axis)
                
                # If still having issues (rare, but possible)
                if axis_norm < 1e-6:
                    world_right = np.array([1.0, 0.0, 0.0])
                    error_axis = np.cross(serv_dock_axis, world_right)
                    axis_norm = np.linalg.norm(error_axis)
        
        # Normalize the axis vector
        error_axis = error_axis / axis_norm
        
        return angle_error, error_axis

    def _calculate_orientation_error(self): # Add more checks
        """Calculates the angular error (in radians) between docking ports' target axes."""
        try:
            serv_mat = self.data.site_xmat[self.site_ids["servicer_dock"]].reshape(3, 3)
            targ_mat = self.data.site_xmat[self.site_ids["target_dock"]].reshape(3, 3)

            if not np.all(np.isfinite(serv_mat)) or not np.all(np.isfinite(targ_mat)):
                 logger.warning(f"Step {self.steps}: NaN/Inf detected in site orientation matrices.")
                 return np.pi

            serv_z_axis_world = serv_mat[:, 2]
            target_alignment_axis_world = -targ_mat[:, 2]

            if not np.all(np.isfinite(serv_z_axis_world)) or not np.all(np.isfinite(target_alignment_axis_world)):
                 logger.warning(f"Step {self.steps}: NaN/Inf detected in derived orientation axes.")
                 return np.pi

            norm_serv = np.linalg.norm(serv_z_axis_world)
            norm_targ = np.linalg.norm(target_alignment_axis_world)
            if norm_serv < 1e-6 or norm_targ < 1e-6:
                 logger.warning(f"Step {self.steps}: Near-zero norm for site orientation axis. ServNorm={norm_serv}, TargNorm={norm_targ}")
                 return np.pi # Treat as max error if axis is undefined

            # Normalize vectors robustly
            serv_z_normed = serv_z_axis_world / norm_serv
            targ_align_normed = target_alignment_axis_world / norm_targ

            dot_prod = np.dot(serv_z_normed, targ_align_normed)
            dot_prod = np.clip(dot_prod, -1.0, 1.0) # Clamp for arccos domain
            angle_rad = np.arccos(dot_prod)

            if not np.isfinite(angle_rad):
                logger.error(f"Step {self.steps}: Non-finite angle ({angle_rad}) calculated in orientation error AFTER arccos. DotProd={dot_prod}. Returning pi.")
                return np.pi
            return float(angle_rad) # Ensure float return

        except IndexError:
             logger.error(f"IndexError calculating orientation error at step {self.steps}. Site IDs might be wrong.")
             return np.pi
        except Exception as e:
            logger.exception(f"Exception calculating orientation error at step {self.steps}: {e}")
            return np.pi


    def _calculate_potential(self, distance, rel_vel_mag, closing_rate,orientation_error): # Add logging/checks
        """Calculates the potential function Φ based on current state. Higher is better."""
        Wd = env_config.POTENTIAL_WEIGHT_DISTANCE
        Wv = env_config.POTENTIAL_WEIGHT_VELOCITY
        Wo = env_config.POTENTIAL_WEIGHT_ORIENT
        EPS = env_config.POTENTIAL_DISTANCE_EPSILON
        # Wd = 0
        # Wv = 0
        # Wo = 0
        # EPS = 0

        # Use safe, clamped values from _get_current_state_metrics
        safe_dist = max(0.01, distance) # Already clamped non-negative
        safe_vel = max(0, rel_vel_mag) # Already clamped non-negative
        safe_orient = max(0, min(np.pi, orientation_error)) # Already clamped 0..pi
        potential_dist = 0.0
        orient_err, orient_guide_axis = self._calculate_orientation_guidance()
    
            # Basic reward inversely proportional to orientation error
            # Convert from [0,π] to [1,0] range
        align = max(0, 1.0 - orient_err / np.pi)

        # --- CHANGE START: Use Linear Negative Distance Potential ---
        potential_dist = Wd * np.exp(-safe_dist / 1.5)
        if not np.isfinite(potential_dist):
             logger.error(f"Potential calc: Non-finite distance potential {potential_dist} (Wd={Wd}, Dist={safe_dist}). Using 0.")
             potential_dist = 0.0
        # --- CHANGE END ---

        '''potential_vel = -Wv * safe_vel
        if not np.isfinite(potential_vel):
             logger.error(f"Potential calc: Non-finite velocity potential {potential_vel} (Wv={Wv}, Vel={safe_vel}). Using 0.")
             potential_vel = 0.0'''
        potential_vel =  Wv * np.tanh(closing_rate)  # Wv > 0 encourages approach
        if not np.isfinite(potential_vel):
            logger.error(f"Potential calc: Non-finite velocity potential {potential_vel} (Wv={Wv}, ClosingRate={safe_closing_rate}). Using 0.")
            potential_vel = 0.0

        potential_orient = -Wo * align
        if not np.isfinite(potential_orient):
             logger.error(f"Potential calc: Non-finite orientation potential {potential_orient} (Wo={Wo}, Orient={safe_orient}). Using 0.")
             potential_orient = 0.0

        potential = potential_dist + potential_vel + potential_orient
        # Potential can be very negative if far away, but shouldn't explode positively
        # Clip might still be useful if Wd is very large, keep default clipping range?
        potential = np.clip(potential, -1e6, +1e6) # Keep clipping for safety

        if not np.isfinite(potential):
            logger.error(f"!!! FINAL POTENTIAL Φ={potential} IS NON-FINITE (Step {self.steps}). Components: D={potential_dist}, V={potential_vel}, O={potential_orient}. INPUTS: Dist={distance}, Vel={rel_vel_mag}, Orient={orientation_error}. CLAMPING TO 0 !!!")
            return 0.0 # Return neutral potential on error

        logger.debug(f"Step {self.steps} Potential Calc (Linear): Φ={potential:.4f} (D={potential_dist:.4f}, V={potential_vel:.4f}, O={potential_orient:.4f})")
        return potential


    def _calculate_rewards_and_done(self): # Add more logging
        """
        Calculates rewards using PBRS, handles terminations/truncations, includes action costs.
        Ensures returned rewards/dones are always finite and cover all agents.
        """
        rewards = {a: 0.0 for a in self.possible_agents} # Initialize for ALL agents
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

        # --- 1. Get Current State Metrics (Robust method) ---
        dist, rel_vel_mag,closing_rate, orient_err = self._get_current_state_metrics()
        logger.debug(f"Step {self.steps} Rewards State: Dist={dist:.4f}, RelVel={rel_vel_mag:.4f}, OrientErr={orient_err:.4f}")

        # --- 2. Check Terminal Conditions ---
        docked = (dist < env_config.DOCKING_DISTANCE_THRESHOLD and
                  rel_vel_mag < env_config.DOCKING_VELOCITY_THRESHOLD and
                  orient_err < env_config.DOCKING_ORIENT_THRESHOLD)

        collision = False
        if not docked:
            try:
                servicer_body_id = self.body_ids[env_config.SERVICER_AGENT_ID]
                target_body_id = self.body_ids[env_config.TARGET_AGENT_ID]
                for i in range(self.data.ncon):
                    c = self.data.contact[i]
                    # Check if bodies involved are servicer and target
                    body1 = self.model.geom_bodyid[c.geom1]
                    body2 = self.model.geom_bodyid[c.geom2]
                    is_serv_targ_contact = ({body1, body2} == {servicer_body_id, target_body_id})
                    # Check penetration distance (dist < 0 means penetration)
                    if is_serv_targ_contact and c.dist <= 0.0: # Use <= 0 for definite penetration
                         collision = True
                         logger.info(f"Step {self.steps}: Collision detected! Contact {i}, dist={c.dist:.4f}, Bodies=({body1},{body2})")
                         break
            except Exception as e: logger.error(f"Error during collision check at step {self.steps}: {e}")

        out_of_bounds = dist > env_config.OUT_OF_BOUNDS_DISTANCE

        # --- Apply Terminal Rewards & Set Terminations/Status ---
        terminal_reward = 0.0
        status = 'in_progress'
        terminate_episode = False

        if docked:
            status = 'docked'; terminal_reward = env_config.REWARD_DOCKING_SUCCESS; terminate_episode = True
            logger.info(f"Step {self.steps}: Docking Successful!")
        elif collision:
            status = 'collision'; terminal_reward = env_config.REWARD_COLLISION; terminate_episode = True
            logger.info(f"Step {self.steps}: Collision Detected!")
        elif out_of_bounds:
            status = 'out_of_bounds'; terminal_reward = env_config.REWARD_OUT_OF_BOUNDS; terminate_episode = True
            logger.info(f"Step {self.steps}: Out Of Bounds (Dist={dist:.2f}m > {env_config.OUT_OF_BOUNDS_DISTANCE}m)!")

        if terminate_episode:
            for a in self.possible_agents: terminations[a] = True

        # Assign terminal rewards (shared)
        for agent in self.possible_agents:
            rewards[agent] += terminal_reward
            if status != 'in_progress': infos[agent]['status'] = status

        # --- 3. Check Truncation Condition ---
        is_truncated = False
        if self.steps >= env_config.MAX_STEPS_PER_EPISODE:
            logger.info(f"Step {self.steps}: Max steps ({env_config.MAX_STEPS_PER_EPISODE}) reached, truncating.")
            status = 'max_steps'
            is_truncated = True
            for agent in self.possible_agents:
                if not terminations.get(agent, False): # Only truncate if not already terminated
                    truncations[agent] = True
                    # Small timeout penalty if not docked
                    if not docked: rewards[agent] -= 500.0
                infos[agent]['status'] = status # Update status to max_steps

        episode_over = terminate_episode or is_truncated

        # --- 4. Calculate Shaping Rewards (PBRS + Action Cost) - Only if episode NOT over ---
        if not episode_over:
            # --- PBRS Calculation (Uses robust _calculate_potential) ---
            current_potential_servicer = self._calculate_potential(dist, rel_vel_mag,closing_rate, orient_err)
            logger.debug(f"Step {self.steps}: Potential Φ(s')_serv = {current_potential_servicer:.4f}, Prev Φ(s)_serv = {self.prev_potential_servicer:.4f}")

            gamma = env_config.POTENTIAL_GAMMA
            # Ensure previous potential is finite (might be inf from reset if start was bad)
            safe_prev_potential = np.nan_to_num(self.prev_potential_servicer, nan=0.0, posinf=0.0, neginf=0.0)

            shaping_reward_servicer = gamma * current_potential_servicer - safe_prev_potential
            # --- CHANGE START: Add Direct Clipping to Shaping Reward ---
                # Define a maximum magnitude for the shaping reward per step to prevent explosions
            MAX_SHAPING_REWARD_MAGNITUDE = 5000.0 # Tune this value if needed (e.g., 20, 50, 100)
            clipped_shaping_reward = np.clip(shaping_reward_servicer, -MAX_SHAPING_REWARD_MAGNITUDE, MAX_SHAPING_REWARD_MAGNITUDE)

                # Optional: Log when clipping actually happens
            if clipped_shaping_reward != shaping_reward_servicer:
                 logger.warning(f"Step {self.steps}: Clipped PBRS reward from {shaping_reward_servicer:.4f} to {clipped_shaping_reward:.4f} (Potential: current={current_potential_servicer:.2f}, prev={safe_prev_potential:.2f})")

            shaping_reward_servicer = clipped_shaping_reward # Use the clipped value
                # --- CHANGE END ---
            # --- Inside your reward calc, after PBRS shaping ---
            '''docking_bonus_active = 5.0  # Activate bonus when within 15 cm
            docking_bonus_scale = 5.0  # Strength of bonus (tune if needed)

            if dist < docking_bonus_active:
                bonus = docking_bonus_scale * (docking_bonus_active - dist)
                shaping_reward_servicer += bonus
                logger.debug(f"Step {self.steps}: Applied docking proximity bonus {bonus:.2f} (dist={dist:.3f})")

            # --- Docking Alignment Bonus ---
            alignment_bonus_active = 2.0  # radians (about 11 degrees)
            alignment_bonus_scale = 3.0  # strength of bonus
            docking_alignment_bonus_active = 0.15  # meters (15 cm)


            if orient_err < alignment_bonus_active and dist < docking_alignment_bonus_active:
                bonus = alignment_bonus_scale * (alignment_bonus_active - orient_err)
                shaping_reward_servicer += bonus
                logger.debug(f"Step {self.steps}: Applied alignment bonus {bonus:.2f} (orient_err={orient_err:.3f})")'''
            serv_ang_vel = self.data.qvel[self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]+3:
                             self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]+6]
            targ_ang_vel = self.data.qvel[self.joint_qvel_adr[env_config.TARGET_AGENT_ID]+3:
                                        self.joint_qvel_adr[env_config.TARGET_AGENT_ID]+6]

            serv_ang_vel_mag = np.linalg.norm(serv_ang_vel)
            targ_ang_vel_mag = np.linalg.norm(targ_ang_vel)

            # Add strong penalties for excessive angular velocities
            ang_vel_penalty_coefficient = 30.0  # Adjust as needed
            max_acceptable_ang_vel = 0.5  # Adjust as needed

            if serv_ang_vel_mag > max_acceptable_ang_vel:
                ang_vel_penalty = -ang_vel_penalty_coefficient * (serv_ang_vel_mag - max_acceptable_ang_vel)
                shaping_reward_servicer += ang_vel_penalty
                logger.debug(f"Angular velocity penalty for servicer: {ang_vel_penalty:.4f}")

            if targ_ang_vel_mag > max_acceptable_ang_vel:
                ang_vel_penalty = -ang_vel_penalty_coefficient * (targ_ang_vel_mag - max_acceptable_ang_vel)
                shaping_reward_servicer += ang_vel_penalty
                logger.debug(f"Angular velocity penalty for target: {ang_vel_penalty:.4f}")

            if not np.isfinite(shaping_reward_servicer):
                logger.error(f"!!! Step {self.steps}: Non-finite PBRS reward ({shaping_reward_servicer}) calculated. γ={gamma}, Φ(s')={current_potential_servicer}, Φ(s)={safe_prev_potential}. Setting to 0. !!!")
                shaping_reward_servicer = 0.0
            

            rewards[env_config.SERVICER_AGENT_ID] += shaping_reward_servicer
            logger.debug(f"Step {self.steps}: Shaping Reward (Servicer) = {shaping_reward_servicer:.4f}")

            # --- Action Cost Penalty ---
            for agent_id in self.possible_agents:
                 action_cost_weight = env_config.REWARD_WEIGHT_ACTION_COST # Default
                 if env_config.COMPETITIVE_MODE: pass # Add competitive logic later if needed

                 if action_cost_weight != 0 and hasattr(self, "current_actions"):
                      action = np.asarray(self.current_actions.get(agent_id, np.zeros(env_config.ACTION_DIM_PER_AGENT)), dtype=np.float64)
                      # Ensure action is finite before squaring
                      if not np.all(np.isfinite(action)):
                           logger.warning(f"Step {self.steps}: Non-finite action {action} for {agent_id} before action cost calc. Skipping cost.")
                           continue
                      action_norm_sq = np.sum(action**2)
                      if np.isfinite(action_norm_sq): # Should be finite if action was
                           action_cost_penalty = action_cost_weight * action_norm_sq
                           rewards[agent_id] += action_cost_penalty
                           logger.debug(f"Step {self.steps}: Action Cost ({agent_id}) = {action_cost_penalty:.6f} (Weight={action_cost_weight})")
                      else:
                           logger.error(f"Step {self.steps}: Non-finite action norm squared ({action_norm_sq}) for {agent_id}. Action={action}")

            # --- Collaborative Mode: Share PBRS Shaping Reward ---
            if not env_config.COMPETITIVE_MODE:
                 rewards[env_config.TARGET_AGENT_ID] += shaping_reward_servicer # Share PBRS
                 logger.debug(f"Step {self.steps}: Shared Shaping Reward (Target) = {shaping_reward_servicer:.4f}")
            else:
                 # Competitive logic placeholder
                 pass

        # --- 5. Update Previous State for Next Step's PBRS ---
        # Use the potential calculated in this step (based on s') as the prev potential for next step
        current_potential_servicer_end = self._calculate_potential(dist, rel_vel_mag,closing_rate, orient_err)
        self.prev_potential_servicer = current_potential_servicer_end # Already checked for finite

        if env_config.COMPETITIVE_MODE:
             # Update prev_potential_target here if needed
             pass

        self.prev_docking_distance = dist # Already checked for finite

        # --- 6. Final Reward Check & Augment Infos ---
        for agent, reward in rewards.items():
            if not np.isfinite(reward):
                logger.error(f"!!! FINAL REWARD FOR {agent} IS NON-FINITE ({reward}) at step {self.steps} BEFORE RETURN !!! Setting to -50.")
                rewards[agent] = -50.0
            # Store final step reward in info (ensure dict exists)
            if agent not in infos: infos[agent] = {}
            infos[agent]['reward_step_total'] = rewards[agent]

        logger.debug(f"Step {self.steps} Final Raw Rewards Returned: {rewards}")
        logger.debug(f"Step {self.steps} Final Raw Terminations: {terminations}")
        logger.debug(f"Step {self.steps} Final Raw Truncations: {truncations}")

        return rewards, terminations, truncations, infos

    # Add the get_reward_info function (Keep as is)
    def get_reward_info(self):
        dist, rel_vel_mag,closing_rate, orient_err = self._get_current_state_metrics()
        current_potential = self._calculate_potential(dist, rel_vel_mag,closing_rate, orient_err)
        reward_info = {
            "state_metrics": {"distance": dist,"relative_velocity": rel_vel_mag,"orientation_error": orient_err,},
            "potential": {"current": current_potential,"previous": self.prev_potential_servicer,"gamma": env_config.POTENTIAL_GAMMA,},
            "thresholds": {"docking_distance": env_config.DOCKING_DISTANCE_THRESHOLD,"docking_velocity": env_config.DOCKING_VELOCITY_THRESHOLD,"docking_orientation": env_config.DOCKING_ORIENT_THRESHOLD,"out_of_bounds": env_config.OUT_OF_BOUNDS_DISTANCE,},
            "reward_values": {"docking_success": env_config.REWARD_DOCKING_SUCCESS,"collision": env_config.REWARD_COLLISION,"out_of_bounds": env_config.REWARD_OUT_OF_BOUNDS,"action_cost_weight": env_config.REWARD_WEIGHT_ACTION_COST,}
        }
        return reward_info

    def get_docking_distance(self):
        servicer_site_pos = self.data.site_xpos[self.site_ids["servicer_dock"]]
        target_site_pos = self.data.site_xpos[self.site_ids["target_dock"]]
        distance = np.linalg.norm(servicer_site_pos - target_site_pos)
        return distance
    def _get_obs(self, agent): # Add more internal checks
        """Calculates the observation vector for the specified agent. Returns zero vector on failure."""
        try:
            servicer_qpos_adr = self.joint_qpos_adr[env_config.SERVICER_AGENT_ID]
            servicer_qvel_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            target_qpos_adr = self.joint_qpos_adr[env_config.TARGET_AGENT_ID]
            target_qvel_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]
            p_s = self.data.site_xpos[self.site_ids["servicer_dock"]]
            p_t = self.data.site_xpos[self.site_ids["target_dock"]]
            if not np.all(np.isfinite(p_s)) or not np.all(np.isfinite(p_t)):
                logger.warning(f"Step {self.steps}: NaN/Inf in site positions for distance calc. Serv={p_s}, Targ={p_t}. Using max distance.")
                dock_dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2 # Use large finite value
            else:
                dock_dist = float(np.linalg.norm(p_t - p_s))
                if not np.isfinite(dock_dist): # Check norm result itself
                     logger.warning(f"Step {self.steps}: Non-finite distance calculated ({dist}). Using max.")
                     dock_dist = env_config.OUT_OF_BOUNDS_DISTANCE * 2

            # --- Read raw data & Perform Robust NaN/Inf checks ---
            data_arrays = {
                "servicer_pos": self.data.qpos[servicer_qpos_adr : servicer_qpos_adr+3],
                "servicer_quat": self.data.qpos[servicer_qpos_adr+3 : servicer_qpos_adr+7],
                "servicer_vel": self.data.qvel[servicer_qvel_adr : servicer_qvel_adr+3],
                "servicer_ang_vel": self.data.qvel[servicer_qvel_adr+3 : servicer_qvel_adr+6],
                "target_pos": self.data.qpos[target_qpos_adr : target_qpos_adr+3],
                "target_quat": self.data.qpos[target_qpos_adr+3 : target_qpos_adr+7],
                "target_vel": self.data.qvel[target_qvel_adr : target_qvel_adr+3],
                "target_ang_vel": self.data.qvel[target_qvel_adr+3 : target_qvel_adr+6],
            }
            corrected_data = {}
            needs_correction = False
            for name, arr in data_arrays.items():
                 original_arr = arr.copy() # Keep original for comparison if needed
                 if not np.all(np.isfinite(arr)):
                     logger.warning(f"NaN/Inf detected in raw MuJoCo data '{name}' for agent {agent} at step {self.steps}: {arr}. Correcting.")
                     needs_correction = True
                     if name.endswith("quat"):
                          # Reset invalid quaternion to default identity [1, 0, 0, 0]
                          arr = np.array([1.0, 0.0, 0.0, 0.0])
                     else:
                          # Clamp positions/velocities using obs clipping bounds for consistency
                          arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=env_config.OUT_OF_BOUNDS_DISTANCE*2, neginf=-env_config.OUT_OF_BOUNDS_DISTANCE*2) # Clamp large positions/vels
                 corrected_data[name] = arr

            # Normalize quaternions AFTER potential correction/reset
            for name in ["servicer_quat", "target_quat"]:
                 quat = corrected_data[name]
                 norm = np.linalg.norm(quat)
                 if norm > 1e-6:
                      corrected_data[name] = quat / norm
                 else:
                      logger.warning(f"Quaternion '{name}' had near-zero norm ({norm}) after potential correction. Resetting to identity.")
                      corrected_data[name] = np.array([1.0, 0.0, 0.0, 0.0])

            if needs_correction: logger.warning(f"MuJoCo data corrected before obs calculation for agent {agent} step {self.steps}.")
            # --- End checks ---

            # Calculate relative state (World Frame) using corrected data
            relative_pos_world = corrected_data["target_pos"] - corrected_data["servicer_pos"]
            relative_vel_world = corrected_data["target_vel"] - corrected_data["servicer_vel"]
            relatice_quat_world = corrected_data["target_quat"] - corrected_data["servicer_quat"]
            relative_ang_vel_world = corrected_data["target_ang_vel"] - corrected_data["servicer_ang_vel"]
            target_ang_vel_world = corrected_data["target_ang_vel"]
            servicer_ang_vel_world = corrected_data["servicer_ang_vel"]
            orient_err, orient_guide_axis = self._calculate_orientation_guidance()

            # Assemble observation list using corrected data
            if agent == env_config.SERVICER_AGENT_ID:
                obs_list = [ relative_pos_world, relative_vel_world,
                             relatice_quat_world, servicer_ang_vel_world,np.array([dock_dist,orient_err]),orient_guide_axis ]
            elif agent == env_config.TARGET_AGENT_ID:
                obs_list = [ -relative_pos_world, -relative_vel_world,
                             -relatice_quat_world, target_ang_vel_world,np.array([dock_dist,orient_err]),orient_guide_axis ]
            else:
                logger.error(f"Unknown agent ID '{agent}' requested for observation.")
                return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)

            # Check components *before* concatenation
            for i, arr in enumerate(obs_list):
                if not np.all(np.isfinite(arr)):
                    logger.error(f"!!! NaN/Inf detected in obs component {i} for agent {agent} BEFORE concatenation (Step {self.steps}): {arr}. Correcting to zero.")
                    obs_list[i] = np.zeros_like(arr)

            # Concatenate into a single observation vector
            obs = np.concatenate(obs_list)

            # Final shape check
            if obs.shape[0] != env_config.OBS_DIM_PER_AGENT:
                 logger.error(f"FATAL Obs dim mismatch for {agent} at step {self.steps}: Got {obs.shape[0]}, expected {env_config.OBS_DIM_PER_AGENT}. Concatenated from: {[c.shape for c in obs_list]}. Returning zero vector.")
                 return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)

            # Final finite check
            if not np.all(np.isfinite(obs)):
                logger.error(f"!!! NaN/Inf detected in FINAL assembled observation for {agent} at step {self.steps}. Obs: {obs}. Returning zero vector.")
                return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)

            # Return correct type
            return obs.astype(np.float32)

        except IndexError:
             logger.exception(f"IndexError calculating observation for agent {agent} at step {self.steps}. MuJoCo IDs/addresses might be wrong.")
             return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)
        except Exception as e:
            logger.exception(f"CRITICAL Error calculating observation for agent {agent} at step {self.steps}: {e}")
            return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)

    def _apply_actions(self, actions): # Keep robust checks
        """Applies actions (scaled forces/torques) to the simulation using xfrc_applied."""
        try:
             if hasattr(self.data, 'xfrc_applied'):
                  self.data.xfrc_applied *= 0.0
             else:
                  raise AttributeError("MuJoCo data object missing 'xfrc_applied'")

             for agent, action in actions.items():
                  if agent not in self.agents: continue

                  body_id = self.body_ids.get(agent)
                  if body_id is None or body_id <= 0:
                       logger.warning(f"Agent {agent} has invalid body ID {body_id}. Cannot apply action.")
                       continue

                  body_row_index = body_id - 1 # xfrc_applied is 0-indexed and excludes world body 0
                  if not (0 <= body_row_index < self.data.xfrc_applied.shape[0]):
                       logger.error(f"Body index {body_row_index} (from body ID {body_id}) is out of bounds for xfrc_applied (shape {self.data.xfrc_applied.shape}).")
                       continue

                  action = np.asarray(action, dtype=np.float64)
                  if action.shape != (env_config.ACTION_DIM_PER_AGENT,):
                       logger.warning(f"Action dim mismatch for agent {agent} at step {self.steps}: Got {action.shape}, expected {(env_config.ACTION_DIM_PER_AGENT,)}. Applying zero.")
                       self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                       continue
                  if not np.all(np.isfinite(action)):
                       logger.warning(f"NaN or Inf detected in raw action for agent {agent} at step {self.steps}: {action}. Applying zero.")
                       self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                       continue

                  force_scale = getattr(env_config, f"ACTION_FORCE_SCALING_{agent.upper()}", env_config.ACTION_FORCE_SCALING)
                  torque_scale = getattr(env_config, f"ACTION_TORQUE_SCALING_{agent.upper()}", env_config.ACTION_TORQUE_SCALING)
                  force = action[:3] * force_scale
                  torque = action[3:] * torque_scale

                  force_torque_6d = np.concatenate([force, torque])

                  if not np.all(np.isfinite(force_torque_6d)):
                       logger.error(f"NaN or Inf detected in *SCALED* action for agent {agent} at step {self.steps}. RawAction={action}, Scaled={force_torque_6d}. Applying zero.")
                       self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                       continue

                  # Apply the calculated force and torque
                  self.data.xfrc_applied[body_row_index, :] = force_torque_6d
                  # logger.debug(f"Applied F/T to {agent} (idx {body_row_index}): {force_torque_6d}")

        except AttributeError as e: # Catch missing xfrc_applied specifically
             logger.error(f"Action application failed: {e}")
             raise # Re-raise as it's fundamental
        except Exception as e:
            logger.exception(f"Unexpected error in _apply_actions at step {self.steps}: {e}")
            # Don't necessarily raise here, but log it. Failure will likely occur in mj_step anyway.


    def render(self): # Keep as is
        """Renders the environment based on the render_mode."""
        if self.render_mode is None:
            logger.warning("Render() called with render_mode=None.", once=True) # Log only once
            return None
        elif self.render_mode == "human":
            if HAS_MEDIAPY:
                 if self.renderer is None:
                      try:
                           self.renderer = mujoco.Renderer(self.model, height=env_config.RENDER_HEIGHT, width=env_config.RENDER_WIDTH)
                           logger.info("MuJoCo Renderer initialized for human mode.")
                      except Exception as e:
                           logger.error(f"Render init error (human): {e}. Cannot render."); self.render_mode = None; return None
                 try:
                      mujoco.mj_forward(self.model, self.data) # Ensure latest state
                      self.renderer.update_scene(self.data, camera="fixed_side")
                      pixels = self.renderer.render()
                      if pixels is not None: media.show_image(pixels); time.sleep(1.0 / self.metadata["render_fps"]); return None
                      else: logger.warning("MuJoCo renderer returned None (human)."); return None
                 except Exception as e: logger.error(f"Human rendering error: {e}"); return None
            else:
                 logger.warning("Human rendering requires 'mediapy'. Install with 'pip install mediapy'. Rendering disabled.", once=True)
                 self.render_mode = None; return None
        elif self.render_mode == "rgb_array":
            return self._render_frame() # Calls helper
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _render_frame(self): # Keep as is
        """Helper to render a single frame as an RGB array."""
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.model, height=env_config.RENDER_HEIGHT, width=env_config.RENDER_WIDTH)
                logger.info("MuJoCo Renderer initialized for rgb_array.")
            except Exception as e:
                 logger.error(f"Error initializing MuJoCo Renderer for rgb_array: {e}. Returning black frame."); self.render_mode = None
                 return np.zeros((env_config.RENDER_HEIGHT, env_config.RENDER_WIDTH, 3), dtype=np.uint8)
        try:
            mujoco.mj_forward(self.model, self.data) # Ensure latest state
            self.renderer.update_scene(self.data, camera="fixed_side")
            pixels = self.renderer.render()
            if pixels is not None: return pixels.astype(np.uint8)
            else: logger.warning("MuJoCo renderer returned None for rgb_array frame."); return np.zeros((env_config.RENDER_HEIGHT, env_config.RENDER_WIDTH, 3), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error during frame rendering for rgb_array: {e}.")
            return np.zeros((env_config.RENDER_HEIGHT, env_config.RENDER_WIDTH, 3), dtype=np.uint8)


    def close(self): # Keep as is
        """Closes the environment and releases resources."""
        if self.renderer:
            try: self.renderer.close()
            except Exception as e: logger.error(f"Error closing MuJoCo renderer: {e}")
            self.renderer = None
        logger.info("SatelliteMARLEnv closed.")