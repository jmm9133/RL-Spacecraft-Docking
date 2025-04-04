# src/satellite_marl_env.py
import gymnasium as gym # Use Gymnasium instead of gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import platform
import time
import logging # Import logging

# Use PettingZoo API
from pettingzoo import ParallelEnv
# from pettingzoo.utils import parallel_to_aec, wrappers # Not using AEC wrappers directly
# from pettingzoo.utils.conversions import parallel_wrapper_fn

# Import configuration
from . import config as env_config # Use alias

# Get a logger for this module
# Ensure logger is configured in the main script (train_marl.py)
# If run standalone, basicConfig might be needed here.
logger = logging.getLogger(__name__)

# Conditional rendering setup
try:
    import mediapy as media
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False
    try:
        import mujoco.viewer
        HAS_MJ_VIEWER = True
    except ImportError:
        HAS_MJ_VIEWER = False


def env(**kwargs):
    """
    The env function wraps the environment in AEC wrappers (not used by default here).
    Provided for potential PettingZoo standard usage elsewhere.
    """
    env = raw_env(**kwargs)
    # from pettingzoo.utils import parallel_to_aec # Import only if needed
    # env = parallel_to_aec(env)
    return env

def raw_env(render_mode=None, **kwargs):
    """
    Instantiates the raw Parallel PettingZoo environment.
    This is the function typically used by the RLlib wrapper creator.
    """
    env = SatelliteMARLEnv(render_mode=render_mode, **kwargs)
    return env


class SatelliteMARLEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "satellite_docking_marl_v0",
        "render_fps": env_config.RENDER_FPS,
    }

    def __init__(self, render_mode=None, **kwargs): # Accept kwargs
        super().__init__()

        xml_path = os.path.abspath(env_config.XML_FILE_PATH)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")
        try:
             self.model = mujoco.MjModel.from_xml_path(xml_path)
             self.data = mujoco.MjData(self.model)
        except Exception as e:
             logger.exception(f"Error loading MuJoCo model from {xml_path}: {e}")
             raise

        self.possible_agents = env_config.POSSIBLE_AGENTS[:]
        self.agent_name_mapping = {i: agent for i, agent in enumerate(self.possible_agents)}
        self.render_mode = render_mode

        # Define Spaces using functions as required by PettingZoo
        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(env_config.OBS_DIM_PER_AGENT,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(env_config.ACTION_DIM_PER_AGENT,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._get_mujoco_ids()

        self.renderer = None
        self.render_frames = []

        self.agents = []
        self.steps = 0
        # Initialize current_actions attribute
        self.current_actions = {}


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

    def reset(self, seed=None, options=None):
        if seed is not None:
             np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # --- Set Initial Conditions ---
        qpos_serv_start = self.joint_qpos_adr[env_config.SERVICER_AGENT_ID]
        qvel_serv_start = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
        self.data.qpos[qpos_serv_start:qpos_serv_start+3] = [0, 0, 0]
        self.data.qpos[qpos_serv_start+3:qpos_serv_start+7] = [1, 0, 0, 0]
        self.data.qvel[qvel_serv_start:qvel_serv_start+6] = [0, 0, 0, 0, 0, 0]

        qpos_targ_start = self.joint_qpos_adr[env_config.TARGET_AGENT_ID]
        qvel_targ_start = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]
        # TODO: Implement randomization based on INITIAL_POS_RANGE, INITIAL_VEL_RANGE
        initial_target_pos = np.array([2.0, 0.5, 0.0]) # Default start
        # Example randomization (needs more robust implementation):
        # rel_pos = np.random.uniform(low=env_config.INITIAL_POS_RANGE[0], high=env_config.INITIAL_POS_RANGE[1])
        # initial_target_pos = self.data.qpos[qpos_serv_start:qpos_serv_start+3] + rel_pos
        self.data.qpos[qpos_targ_start:qpos_targ_start+3] = initial_target_pos
        self.data.qpos[qpos_targ_start+3:qpos_targ_start+7] = [1, 0, 0, 0]
        self.data.qvel[qvel_targ_start:qvel_targ_start+6] = [0, 0, 0, 0, 0, 0] # Start stationary

        mujoco.mj_forward(self.model, self.data) # Compute initial state kinematics, contacts

        # --- Reset PettingZoo State ---
        self.agents = self.possible_agents[:]
        self.steps = 0
        self.render_frames = []
        self.current_actions = {} # Clear stored actions

        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        if self.render_mode == "human":
            self.render()

        return observations, infos


    def _get_obs(self, agent):
        try:
            servicer_qpos_adr = self.joint_qpos_adr[env_config.SERVICER_AGENT_ID]
            servicer_qvel_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            target_qpos_adr = self.joint_qpos_adr[env_config.TARGET_AGENT_ID]
            target_qvel_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]

            servicer_pos = self.data.qpos[servicer_qpos_adr : servicer_qpos_adr+3]
            servicer_quat = self.data.qpos[servicer_qpos_adr+3 : servicer_qpos_adr+7]
            servicer_vel = self.data.qvel[servicer_qvel_adr : servicer_qvel_adr+3]
            servicer_ang_vel = self.data.qvel[servicer_qvel_adr+3 : servicer_qvel_adr+6]

            target_pos = self.data.qpos[target_qpos_adr : target_qpos_adr+3]
            target_quat = self.data.qpos[target_qpos_adr+3 : target_qpos_adr+7]
            target_vel = self.data.qvel[target_qvel_adr : target_qvel_adr+3]
            target_ang_vel = self.data.qvel[target_qvel_adr+3 : target_qvel_adr+6]

            relative_pos_world = target_pos - servicer_pos
            relative_vel_world = target_vel - servicer_vel

            if agent == env_config.SERVICER_AGENT_ID:
                obs = np.concatenate([ relative_pos_world, relative_vel_world,
                                       servicer_quat, servicer_ang_vel ]).astype(np.float32)
            elif agent == env_config.TARGET_AGENT_ID:
                obs = np.concatenate([ -relative_pos_world, -relative_vel_world,
                                       target_quat, target_ang_vel ]).astype(np.float32)
            else:
                raise ValueError(f"Unknown agent ID: {agent}")

            if obs.shape[0] != env_config.OBS_DIM_PER_AGENT:
                 logger.warning(f"Obs dim mismatch {agent}: {obs.shape[0]} vs {env_config.OBS_DIM_PER_AGENT}")
                 # Basic handling - pad or truncate
                 diff = env_config.OBS_DIM_PER_AGENT - obs.shape[0]
                 if diff > 0: obs = np.pad(obs, (0, diff), 'constant')
                 elif diff < 0: obs = obs[:env_config.OBS_DIM_PER_AGENT]

            # Check for NaN/Inf values - replace with zeros as a safeguard
            if not np.all(np.isfinite(obs)):
                logger.warning(f"NaN or Inf detected in observation for {agent}. Replacing with zeros.")
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


            return obs

        except Exception as e:
            logger.exception(f"Error calculating observation for agent {agent}: {e}")
            return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)


    def _apply_actions(self, actions):
        """Applies actions to the simulation using xfrc_applied."""
        self.data.xfrc_applied *= 0.0 # Reset forces

        for agent, action in actions.items():
            if agent not in self.agents: continue # Skip inactive agents

            if agent not in self.body_ids:
                logger.warning(f"Agent {agent} has no body ID.")
                continue

            body_id = self.body_ids[agent]
            if body_id <= 0: continue # Skip world body

            body_row_index = body_id - 1
            if body_row_index >= self.data.xfrc_applied.shape[0]:
                 logger.error(f"Body index {body_row_index} OOB for xfrc_applied.")
                 continue

            if len(action) != env_config.ACTION_DIM_PER_AGENT:
                 logger.warning(f"Action dim mismatch for {agent}: {len(action)} vs {env_config.ACTION_DIM_PER_AGENT}")
                 continue

            force = action[:3] * env_config.ACTION_FORCE_SCALING
            torque = action[3:] * env_config.ACTION_TORQUE_SCALING
            force_torque_6d = np.concatenate([force, torque])

            # Check for NaN/Inf in actions before applying
            if not np.all(np.isfinite(force_torque_6d)):
                logger.warning(f"NaN or Inf detected in action for agent {agent}. Applying zero force/torque.")
                force_torque_6d = np.zeros(6)

            try:
                self.data.xfrc_applied[body_row_index, :] = force_torque_6d
            except Exception as e: # Catch potential errors during assignment
                 logger.exception(f"Error assigning xfrc_applied for agent {agent}: {e}")


    def _calculate_rewards_and_done(self):
        """Calculates rewards, terminations, truncations, and infos."""
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        active_agents = self.agents # Use current active agents

        # --- Calculate docking status ---
        is_docked = False
        docking_distance = float('inf')
        relative_velocity_mag = float('inf')
        try:
            servicer_dock_pos = self.data.site_xpos[self.site_ids["servicer_dock"]]
            target_dock_pos = self.data.site_xpos[self.site_ids["target_dock"]]
            docking_distance = np.linalg.norm(servicer_dock_pos - target_dock_pos)

            servicer_qvel_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            target_qvel_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]
            servicer_lin_vel = self.data.qvel[servicer_qvel_adr : servicer_qvel_adr+3]
            target_lin_vel = self.data.qvel[target_qvel_adr : target_qvel_adr+3]
            relative_velocity_mag = np.linalg.norm(servicer_lin_vel - target_lin_vel)

            is_docked = (docking_distance < env_config.DOCKING_DISTANCE_THRESHOLD and
                         relative_velocity_mag < env_config.DOCKING_VELOCITY_THRESHOLD)
        except Exception as e: logger.error(f"Error calculating docking status: {e}")

        # --- Check for collisions ---
        is_collision = False
        try:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]
                servicer_body_id = self.body_ids[env_config.SERVICER_AGENT_ID]
                target_body_id = self.body_ids[env_config.TARGET_AGENT_ID]
                if (geom1_body == servicer_body_id and geom2_body == target_body_id) or \
                   (geom1_body == target_body_id and geom2_body == servicer_body_id):
                    if not is_docked: is_collision = True; break
        except Exception as e: logger.error(f"Error checking collisions: {e}")

        # --- Termination conditions ---
        if is_docked:
            logger.info("Docking Successful!")
            for agent in self.possible_agents: # Terminate all on success
                 terminations[agent] = True
                 rewards[agent] += env_config.REWARD_DOCKING_SUCCESS
                 infos[agent]['status'] = 'docked'

        elif is_collision:
            logger.info("Collision Detected!")
            for agent in self.possible_agents: # Terminate all on collision
                 terminations[agent] = True
                 rewards[agent] += env_config.REWARD_COLLISION
                 infos[agent]['status'] = 'collision'

        # --- Truncation conditions ---
        if self.steps >= env_config.MAX_STEPS_PER_EPISODE:
            logger.info("Max steps reached, episode truncated.")
            for agent in self.possible_agents:
                 if not terminations[agent]: # Only truncate if not already terminated
                      truncations[agent] = True
                      infos[agent]['status'] = 'max_steps'

        # --- Reward Shaping (Applied if episode is not done) ---
        is_terminated_any = any(terminations.values())
        is_truncated_any = any(truncations.values())
        is_done = is_terminated_any or is_truncated_any

        if not is_done:
            # Distance Penalty (Servicer)
            if env_config.REWARD_WEIGHT_DISTANCE != 0 and np.isfinite(docking_distance):
                distance_penalty = docking_distance * env_config.REWARD_WEIGHT_DISTANCE
                if env_config.SERVICER_AGENT_ID in active_agents:
                    rewards[env_config.SERVICER_AGENT_ID] += distance_penalty

            # Velocity Penalty (Servicer)
            if env_config.REWARD_WEIGHT_VELOCITY_MAG != 0 and np.isfinite(relative_velocity_mag):
                 velocity_penalty = relative_velocity_mag * env_config.REWARD_WEIGHT_VELOCITY_MAG
                 if env_config.SERVICER_AGENT_ID in active_agents:
                     rewards[env_config.SERVICER_AGENT_ID] += velocity_penalty

            # Action Cost Penalty (All Active Agents)
            if env_config.REWARD_WEIGHT_ACTION_COST != 0:
                 if hasattr(self, 'current_actions') and self.current_actions:
                     for agent_id, action in self.current_actions.items():
                          if agent_id in active_agents:
                              action_norm = np.linalg.norm(action)
                              if np.isfinite(action_norm): # Avoid adding nan cost
                                   action_magnitude_penalty = action_norm * env_config.REWARD_WEIGHT_ACTION_COST
                                   rewards[agent_id] += action_magnitude_penalty
                 else: logger.warning("Cannot apply action cost: self.current_actions not found/empty.")

        # Final check for NaN/Inf in rewards and replace with 0
        for agent in self.possible_agents:
            if not np.isfinite(rewards[agent]):
                logger.warning(f"NaN or Inf detected in reward for {agent}. Setting reward to 0.")
                rewards[agent] = 0.0

        return rewards, terminations, truncations, infos


    def step(self, actions):
        self.current_actions = actions.copy() # Store actions for reward calc
        active_actions = {agent: actions[agent] for agent in self.agents if agent in actions}

        try: self._apply_actions(active_actions)
        except Exception as e: logger.exception(f"Error applying actions: {e}")

        try:
             mujoco.mj_step(self.model, self.data)
             self.steps += 1
        except Exception as e:
             logger.exception(f"Error during MuJoCo mj_step: {e}")
             # ... (handle mj_step error and return) ...
             rewards = {agent: -100.0 for agent in self.possible_agents}
             terminations = {agent: True for agent in self.possible_agents}
             truncations = {agent: False for agent in self.possible_agents}
             infos = {agent: {'status': 'mj_step_error'} for agent in self.possible_agents}
             observations = {agent: self._get_obs(agent) for agent in self.possible_agents} # Try to get last obs
             self.agents = []
             # Ensure finite values before returning
             for agent_id, obs in observations.items():
                 observations[agent_id] = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
             return observations, rewards, terminations, truncations, infos


        rewards, terminations, truncations, infos = self._calculate_rewards_and_done()
        self.agents = [agent for agent in self.agents if not (terminations.get(agent, False) or truncations.get(agent, False))]
        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}

        # --- **DEBUG LOGGING START** ---
        if self.steps % 200 == 0: # Log every 200 steps to avoid spamming
             logger.debug(f"--- Step {self.steps} Data ---")
             for agent_id in self.possible_agents:
                  if agent_id in observations:
                       obs_agent = observations[agent_id]
                       logger.debug(f"  Agent: {agent_id}")
                       logger.debug(f"    Obs (finite: {np.all(np.isfinite(obs_agent))} | mean: {np.mean(obs_agent):.3f} | std: {np.std(obs_agent):.3f}): {obs_agent}")
                       logger.debug(f"    Reward: {rewards.get(agent_id, 0.0):.3f}")
                       logger.debug(f"    Terminated: {terminations.get(agent_id, False)}")
                       logger.debug(f"    Truncated: {truncations.get(agent_id, False)}")
             logger.debug(f"  Active Agents: {self.agents}")
             logger.debug(f"----------------------")
        # --- **DEBUG LOGGING END** ---


        if self.render_mode == "human": self.render()
        elif self.render_mode == "rgb_array": self._render_frame()

        if hasattr(self, 'current_actions'): del self.current_actions

        # Final check for NaN/Inf in observations just before returning
        for agent_id, obs in observations.items():
            if not np.all(np.isfinite(obs)):
                 logger.error(f"!!! NaN/Inf in FINAL observation for {agent_id} at step {self.steps} !!!")
                 observations[agent_id] = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


        return observations, rewards, terminations, truncations, infos


    def _render_frame(self):
        if self.render_mode != "rgb_array": return
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.model, height=env_config.RENDER_HEIGHT, width=env_config.RENDER_WIDTH)
                logger.info("MuJoCo Renderer initialized for rgb_array.")
            except Exception as e:
                 logger.error(f"Error initializing MuJoCo Renderer: {e}")
                 self.render_mode = None
                 return
        try:
            self.renderer.update_scene(self.data, camera="fixed_side")
            pixels = self.renderer.render()
            if pixels is not None: self.render_frames.append(pixels)
        except Exception as e: logger.error(f"Error during frame rendering: {e}")

    def render(self):
        if self.render_mode == "human":
            if HAS_MEDIAPY:
                 if self.renderer is None:
                      try:
                           self.renderer = mujoco.Renderer(self.model, height=env_config.RENDER_HEIGHT, width=env_config.RENDER_WIDTH)
                           logger.info("MuJoCo Renderer initialized for human mode.")
                      except Exception as e: logger.error(f"Render init error (human): {e}"); return
                 try:
                      self.renderer.update_scene(self.data, camera="fixed_side")
                      pixels = self.renderer.render()
                      if pixels is not None:
                           media.show_image(pixels)
                           time.sleep(1.0 / self.metadata["render_fps"])
                      else: logger.warning("Renderer returned None (human).")
                 except Exception as e: logger.error(f"Human rendering error: {e}")
            else:
                 if not hasattr(self, '_human_render_warned'): # Warn only once
                      logger.warning("Human rendering requires 'mediapy'. Cannot show real-time video.")
                      self._human_render_warned = True
            return None
        elif self.render_mode == "rgb_array":
            self._render_frame()
            if self.render_frames: return self.render_frames[-1]
            else: return np.zeros((env_config.RENDER_HEIGHT, env_config.RENDER_WIDTH, 3), dtype=np.uint8)

    def close(self):
        if self.renderer:
            try: self.renderer.close()
            except Exception as e: logger.error(f"Error closing renderer: {e}")
            self.renderer = None
        logger.info("SatelliteMARLEnv closed.")