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
             logger.info(f"MuJoCo model loaded successfully from {xml_path}")
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
        # Initialize docking distance tracking
        self.prev_docking_distance = float('inf')


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

    def reset(self, seed=None, options=None):
        logger.debug("--- Environment Reset Called ---")
        if seed is not None:
             logger.debug(f"Resetting with seed: {seed}")
             np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        logger.debug("MuJoCo data reset.")

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

        logger.debug(f"Reset: Initial Servicer qpos[0:7]: {self.data.qpos[qpos_serv_start:qpos_serv_start+7]}")
        logger.debug(f"Reset: Initial Target qpos[0:7]: {self.data.qpos[qpos_targ_start:qpos_targ_start+7]}")
        logger.debug(f"Reset: Initial Servicer qvel[0:6]: {self.data.qvel[qvel_serv_start:qvel_serv_start+6]}")
        logger.debug(f"Reset: Initial Target qvel[0:6]: {self.data.qvel[qvel_targ_start:qvel_targ_start+6]}")


        mujoco.mj_forward(self.model, self.data) # Compute initial state kinematics, contacts
        logger.debug("Initial mj_forward() completed.")

        # --- Reset PettingZoo State ---
        self.agents = self.possible_agents[:]
        self.steps = 0
        self.render_frames = []
        self.current_actions = {} # Clear stored actions

        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        # Log initial observations
        for agent, obs in observations.items():
            obs_finite = np.all(np.isfinite(obs))
            logger.debug(f"Reset: Initial Obs '{agent}' (finite: {obs_finite}): {obs}")
            if not obs_finite:
                 logger.error(f"!!! NaN/Inf DETECTED IN RESET OBSERVATION for {agent} !!!")

        # Check initial state conditions for immediate termination/reward issues
        initial_docking_distance = float('nan')
        initial_relative_velocity_mag = float('nan')
        try:
            servicer_dock_pos = self.data.site_xpos[self.site_ids["servicer_dock"]]
            target_dock_pos = self.data.site_xpos[self.site_ids["target_dock"]]
            initial_docking_distance = np.linalg.norm(servicer_dock_pos - target_dock_pos)

            servicer_qvel_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            target_qvel_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]
            servicer_lin_vel = self.data.qvel[servicer_qvel_adr : servicer_qvel_adr+3]
            target_lin_vel = self.data.qvel[target_qvel_adr : target_qvel_adr+3]
            initial_relative_velocity_mag = np.linalg.norm(servicer_lin_vel - target_lin_vel)
            logger.debug(f"Reset: Initial Docking Dist: {initial_docking_distance:.4f}, Rel Vel Mag: {initial_relative_velocity_mag:.4f}")

            # Check for immediate collision at reset
            initial_collision = False
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]
                servicer_body_id = self.body_ids[env_config.SERVICER_AGENT_ID]
                target_body_id = self.body_ids[env_config.TARGET_AGENT_ID]
                # Check if the collision involves the main bodies (ignore potential self-collisions or ground)
                if ((geom1_body == servicer_body_id and geom2_body == target_body_id) or \
                    (geom1_body == target_body_id and geom2_body == servicer_body_id)) and \
                    contact.dist < 0.001: # Ensure actual penetration
                    initial_collision = True; break
            logger.debug(f"Reset: Initial Collision Detected: {initial_collision} (ncon={self.data.ncon})")

        except Exception as e:
            logger.exception(f"Error checking initial state conditions: {e}")

        # Initialize previous distance state for reward calculation
        self.prev_docking_distance = initial_docking_distance if np.isfinite(initial_docking_distance) else 0.0
        logger.debug(f"Reset: Initialized prev_docking_distance = {self.prev_docking_distance:.4f}")

        if self.render_mode == "human":
            self.render()

        logger.debug(f"--- Environment Reset Finished. Active agents: {self.agents} ---")
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

            # --- Check for NaN/Inf in raw MuJoCo data ---
            raw_data_list = [servicer_pos, servicer_quat, servicer_vel, servicer_ang_vel,
                             target_pos, target_quat, target_vel, target_ang_vel]
            if not all(np.all(np.isfinite(arr)) for arr in raw_data_list):
                logger.warning(f"NaN/Inf detected in raw MuJoCo qpos/qvel data for agent {agent} at step {self.steps}.")
                # Attempt to log specific bad data
                for name, arr in zip(["s_pos", "s_quat", "s_vel", "s_ang", "t_pos", "t_quat", "t_vel", "t_ang"], raw_data_list):
                    if not np.all(np.isfinite(arr)): logger.warning(f"  Invalid data in {name}: {arr}")
                # Replace bad arrays with zeros temporarily to avoid crashing observation calculation
                servicer_pos = np.nan_to_num(servicer_pos)
                servicer_quat = np.nan_to_num(servicer_quat) # Note: Quat should be normalized, [0,0,0,0] is invalid
                if not np.all(np.isfinite(servicer_quat)): servicer_quat = np.array([1.0, 0.0, 0.0, 0.0]) # Reset quat
                servicer_vel = np.nan_to_num(servicer_vel)
                servicer_ang_vel = np.nan_to_num(servicer_ang_vel)
                target_pos = np.nan_to_num(target_pos)
                target_quat = np.nan_to_num(target_quat)
                if not np.all(np.isfinite(target_quat)): target_quat = np.array([1.0, 0.0, 0.0, 0.0])
                target_vel = np.nan_to_num(target_vel)
                target_ang_vel = np.nan_to_num(target_ang_vel)
            # --- End Check ---


            relative_pos_world = target_pos - servicer_pos
            relative_vel_world = target_vel - servicer_vel

            if agent == env_config.SERVICER_AGENT_ID:
                obs = np.concatenate([ relative_pos_world, relative_vel_world,
                                       servicer_quat, servicer_ang_vel ])
            elif agent == env_config.TARGET_AGENT_ID:
                # Target observes negative relative pos/vel from its perspective
                obs = np.concatenate([ -relative_pos_world, -relative_vel_world,
                                       target_quat, target_ang_vel ])
            else:
                logger.error(f"Unknown agent ID requested for observation: {agent}")
                raise ValueError(f"Unknown agent ID: {agent}")

            # Ensure correct shape BEFORE casting type
            if obs.shape[0] != env_config.OBS_DIM_PER_AGENT:
                 logger.warning(f"Obs dim mismatch for {agent} at step {self.steps}: Got {obs.shape[0]}, expected {env_config.OBS_DIM_PER_AGENT}. Padding/Truncating.")
                 diff = env_config.OBS_DIM_PER_AGENT - obs.shape[0]
                 if diff > 0: obs = np.pad(obs, (0, diff), 'constant')
                 elif diff < 0: obs = obs[:env_config.OBS_DIM_PER_AGENT]

            # Final check for NaN/Inf values in the assembled observation - replace with zeros as a safeguard
            obs_finite = np.all(np.isfinite(obs))
            if not obs_finite:
                logger.warning(f"NaN or Inf detected in assembled observation for {agent} at step {self.steps}. Replacing with zeros.")
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

            return obs.astype(np.float32) # Cast to float32 at the very end

        except Exception as e:
            logger.exception(f"Error calculating observation for agent {agent} at step {self.steps}: {e}")
            # Return a zero vector of the correct shape and type
            return np.zeros(env_config.OBS_DIM_PER_AGENT, dtype=np.float32)


    def _apply_actions(self, actions):
        """Applies actions to the simulation using xfrc_applied."""
        self.data.xfrc_applied *= 0.0 # Reset forces/torques from previous step

        for agent, action in actions.items():
            if agent not in self.agents:
                # logger.debug(f"Skipping action application for inactive agent: {agent}")
                continue # Skip inactive agents

            # Ensure action is numpy array for safety
            action = np.asarray(action, dtype=np.float64)

            if agent not in self.body_ids:
                logger.warning(f"Agent {agent} has no body ID defined. Cannot apply action.")
                continue

            body_id = self.body_ids[agent]
            if body_id <= 0: # Should not happen if body_ids are correct
                 logger.error(f"Invalid body ID {body_id} for agent {agent}.")
                 continue

            # MuJoCo body IDs are 1-based, index for xfrc_applied is 0-based (excluding world body 0)
            body_row_index = body_id - 1
            if body_row_index < 0 or body_row_index >= self.data.xfrc_applied.shape[0]:
                 logger.error(f"Body index {body_row_index} (from body ID {body_id}) is out of bounds for xfrc_applied (shape {self.data.xfrc_applied.shape}).")
                 continue

            if action.shape != (env_config.ACTION_DIM_PER_AGENT,):
                 logger.warning(f"Action dim mismatch for agent {agent} at step {self.steps}: Got {action.shape}, expected {(env_config.ACTION_DIM_PER_AGENT,)}. Applying zero force.")
                 # Apply zero force instead of potentially crashing with wrong shape
                 self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                 continue

            # Check for NaN/Inf in actions before scaling/applying
            action_finite = np.all(np.isfinite(action))
            if not action_finite:
                logger.warning(f"NaN or Inf detected in raw action for agent {agent} at step {self.steps}: {action}. Applying zero force/torque.")
                self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                continue

            # Apply scaling
            force = action[:3] * env_config.ACTION_FORCE_SCALING
            torque = action[3:] * env_config.ACTION_TORQUE_SCALING
            force_torque_6d = np.concatenate([force, torque])

            # Double-check scaled actions
            scaled_action_finite = np.all(np.isfinite(force_torque_6d))
            if not scaled_action_finite:
                 logger.error(f"NaN or Inf detected in *SCALED* action for agent {agent} at step {self.steps}. RawAction={action}, Scaled={force_torque_6d}. Applying zero force/torque.")
                 self.data.xfrc_applied[body_row_index, :] = np.zeros(6)
                 continue

            try:
                # Apply the calculated force and torque
                # logger.debug(f"Applying force/torque to {agent} (body {body_id}): {force_torque_6d}")
                self.data.xfrc_applied[body_row_index, :] = force_torque_6d
            except IndexError:
                 logger.error(f"IndexError assigning xfrc_applied for agent {agent} at index {body_row_index}.")
            except Exception as e: # Catch potential other errors during assignment
                 logger.exception(f"Unexpected error assigning xfrc_applied for agent {agent} at index {body_row_index}: {e}")


    def _calculate_rewards_and_done(self):
        """Calculates rewards, terminations, truncations, and infos."""
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        active_agents = self.agents[:] # Use current active agents list

        # --- Calculate docking status ---
        is_docked = False
        docking_distance = float('inf')
        relative_velocity_mag = float('inf')
        current_docking_distance_finite = False
        current_rel_vel_finite = False

        try:
            servicer_dock_pos = self.data.site_xpos[self.site_ids["servicer_dock"]]
            target_dock_pos = self.data.site_xpos[self.site_ids["target_dock"]]
            # Check if site positions are valid
            if not np.all(np.isfinite(servicer_dock_pos)) or not np.all(np.isfinite(target_dock_pos)):
                 logger.warning(f"NaN/Inf in docking site positions at step {self.steps}. Servicer={servicer_dock_pos}, Target={target_dock_pos}")
            else:
                 docking_distance = np.linalg.norm(servicer_dock_pos - target_dock_pos)
                 current_docking_distance_finite = np.isfinite(docking_distance)

            servicer_qvel_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
            target_qvel_adr = self.joint_qvel_adr[env_config.TARGET_AGENT_ID]
            servicer_lin_vel = self.data.qvel[servicer_qvel_adr : servicer_qvel_adr+3]
            target_lin_vel = self.data.qvel[target_qvel_adr : target_qvel_adr+3]
            # Check if velocities are valid
            if not np.all(np.isfinite(servicer_lin_vel)) or not np.all(np.isfinite(target_lin_vel)):
                logger.warning(f"NaN/Inf in body velocities for docking check at step {self.steps}. Servicer={servicer_lin_vel}, Target={target_lin_vel}")
            else:
                relative_velocity_mag = np.linalg.norm(servicer_lin_vel - target_lin_vel)
                current_rel_vel_finite = np.isfinite(relative_velocity_mag)

            # Check docking success condition only if distances/velocities are valid
            if current_docking_distance_finite and current_rel_vel_finite:
                is_docked = (docking_distance < env_config.DOCKING_DISTANCE_THRESHOLD and
                             relative_velocity_mag < env_config.DOCKING_VELOCITY_THRESHOLD)

            logger.debug(f"CalcRewards Step {self.steps}: DockingDist={docking_distance if current_docking_distance_finite else 'NaN':.4f}, RelVelMag={relative_velocity_mag if current_rel_vel_finite else 'NaN':.4f}, IsDocked={is_docked}")

        except IndexError:
             logger.error(f"IndexError calculating docking status at step {self.steps}. Site or joint IDs might be invalid.")
        except Exception as e:
             logger.exception(f"Unexpected error calculating docking status at step {self.steps}: {e}")

        # --- Check for collisions ---
        is_collision = False
        try:
            num_contacts = self.data.ncon
            servicer_body_id = self.body_ids[env_config.SERVICER_AGENT_ID]
            target_body_id = self.body_ids[env_config.TARGET_AGENT_ID]
            for i in range(num_contacts):
                contact = self.data.contact[i]
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]

                # Check if the collision is between the two main satellite bodies
                is_target_servicer_collision = (geom1_body == servicer_body_id and geom2_body == target_body_id) or \
                                               (geom1_body == target_body_id and geom2_body == servicer_body_id)

                if is_target_servicer_collision and contact.dist < 0.001: # Check for actual penetration
                    # Avoid penalizing collision if successful docking occurred in the same step
                    if not is_docked:
                        is_collision = True
                        logger.debug(f"CalcRewards Step {self.steps}: Collision detected between servicer (body {servicer_body_id}) and target (body {target_body_id}), contact {i}, dist={contact.dist:.4f}")
                        break # Stop checking once a collision is confirmed
            # logger.debug(f"CalcRewards Step {self.steps}: Collision Check Complete. IsCollision={is_collision} (ncon={num_contacts})")

        except IndexError:
            logger.error(f"IndexError checking collisions at step {self.steps}. Body/Geom IDs may be invalid.")
        except Exception as e:
            logger.exception(f"Unexpected error checking collisions at step {self.steps}: {e}")


        # --- Termination conditions ---
        if is_docked:
            logger.info(f"Docking Successful at step {self.steps}!")
            for agent in self.possible_agents: # Terminate all on success
                 terminations[agent] = True
                 rewards[agent] += env_config.REWARD_DOCKING_SUCCESS
                 if agent in infos: infos[agent]['status'] = 'docked'
                 else: infos[agent] = {'status': 'docked'}

        elif is_collision:
            logger.info(f"Collision Detected at step {self.steps}!")
            for agent in self.possible_agents: # Terminate all on collision
                 terminations[agent] = True
                 rewards[agent] += env_config.REWARD_COLLISION
                 if agent in infos: infos[agent]['status'] = 'collision'
                 else: infos[agent] = {'status': 'collision'}

        # --- Truncation conditions ---
        if self.steps >= env_config.MAX_STEPS_PER_EPISODE:
            logger.info(f"Max steps ({env_config.MAX_STEPS_PER_EPISODE}) reached, episode truncated at step {self.steps}.")
            for agent in self.possible_agents:
                 if not terminations.get(agent, False): # Only truncate if not already terminated
                      truncations[agent] = True
                      if agent in infos: infos[agent]['status'] = 'max_steps'
                      else: infos[agent] = {'status': 'max_steps'}

        # --- Reward Shaping (Applied ONLY if episode is not done/terminated/truncated yet) ---
        is_terminated_this_step = any(terminations.values())
        is_truncated_this_step = any(truncations.values())
        episode_is_over = is_terminated_this_step or is_truncated_this_step
        logger.debug(f"CalcRewards Step {self.steps}: Episode is Over: {episode_is_over} (Term={is_terminated_this_step}, Trunc={is_truncated_this_step})")


        # Store intermediate reward components for logging
        reward_component_log = {agent: {'base': rewards[agent]} for agent in self.possible_agents}

        if not episode_is_over:
            logger.debug(f"CalcRewards Step {self.steps}: Applying shaping rewards.")
            # --- Distance Delta Reward (Servicer) ---
            dist_delta_reward = 0.0
            # Use the valid distance from this step, or fall back if it was invalid
            current_dist_for_delta = docking_distance if current_docking_distance_finite else self.prev_docking_distance
            # Ensure previous distance was valid too
            if np.isfinite(self.prev_docking_distance) and current_docking_distance_finite:
                dist_delta = self.prev_docking_distance - current_dist_for_delta # Positive if got closer
                dist_delta_reward = env_config.REWARD_WEIGHT_DISTANCE_DELTA * dist_delta
                logger.debug(f"CalcRewards Step {self.steps}: PrevDist={self.prev_docking_distance:.4f}, CurrDist={current_dist_for_delta:.4f}, Delta={dist_delta:.4f}, DeltaReward={dist_delta_reward:.4f}")
                if env_config.SERVICER_AGENT_ID in active_agents:
                    rewards[env_config.SERVICER_AGENT_ID] += dist_delta_reward
                    reward_component_log[env_config.SERVICER_AGENT_ID]['dist_delta'] = dist_delta_reward
            else:
                 logger.warning(f"CalcRewards Step {self.steps}: Skipping distance delta reward due to non-finite distance (Prev={self.prev_docking_distance}, Curr={docking_distance}).")

            # --- Distance Penalty (Servicer) ---
            dist_penalty = 0.0
            if env_config.REWARD_WEIGHT_DISTANCE != 0 and current_docking_distance_finite:
                dist_penalty = docking_distance * env_config.REWARD_WEIGHT_DISTANCE
                logger.debug(f"CalcRewards Step {self.steps}: DistPenalty={dist_penalty:.4f}")
                if env_config.SERVICER_AGENT_ID in active_agents:
                    rewards[env_config.SERVICER_AGENT_ID] += dist_penalty
                    reward_component_log[env_config.SERVICER_AGENT_ID]['dist_penalty'] = dist_penalty
            elif env_config.REWARD_WEIGHT_DISTANCE != 0:
                 logger.warning(f"CalcRewards Step {self.steps}: Skipping distance penalty due to non-finite distance.")


            # --- Velocity Penalty (Servicer) ---
            vel_penalty = 0.0
            if env_config.REWARD_WEIGHT_VELOCITY_MAG != 0 and current_rel_vel_finite:
                 vel_penalty = relative_velocity_mag * env_config.REWEIGHT_VELOCITY_MAG
                 logger.debug(f"CalcRewards Step {self.steps}: VelPenalty={vel_penalty:.4f}")
                 if env_config.SERVICER_AGENT_ID in active_agents:
                     rewards[env_config.SERVICER_AGENT_ID] += vel_penalty
                     reward_component_log[env_config.SERVICER_AGENT_ID]['vel_penalty'] = vel_penalty
            elif env_config.REWARD_WEIGHT_VELOCITY_MAG != 0:
                 logger.warning(f"CalcRewards Step {self.steps}: Skipping velocity penalty due to non-finite relative velocity.")


            # --- Action Cost Penalty (All Active Agents) ---
            action_magnitude_penalty_serv = 0.0
            action_magnitude_penalty_targ = 0.0
            if env_config.REWARD_WEIGHT_ACTION_COST != 0:
                 if hasattr(self, 'current_actions') and self.current_actions:
                     for agent_id, action in self.current_actions.items():
                          if agent_id in active_agents:
                              action_np = np.asarray(action) # Ensure numpy
                              action_norm = np.linalg.norm(action_np)
                              if np.isfinite(action_norm): # Avoid adding nan cost
                                   action_magnitude_penalty = action_norm * env_config.REWARD_WEIGHT_ACTION_COST
                                   rewards[agent_id] += action_magnitude_penalty
                                   if agent_id == env_config.SERVICER_AGENT_ID:
                                       action_magnitude_penalty_serv = action_magnitude_penalty
                                       reward_component_log[agent_id]['action_cost'] = action_magnitude_penalty
                                   elif agent_id == env_config.TARGET_AGENT_ID:
                                       action_magnitude_penalty_targ = action_magnitude_penalty
                                       reward_component_log[agent_id]['action_cost'] = action_magnitude_penalty
                              else:
                                   logger.warning(f"CalcRewards Step {self.steps}: NaN/Inf Action norm for {agent_id}. Action={action_np}")
                     logger.debug(f"CalcRewards Step {self.steps}: ActionCostServ={action_magnitude_penalty_serv:.4f}, ActionCostTarg={action_magnitude_penalty_targ:.4f}")
                 else:
                      logger.warning(f"CalcRewards Step {self.steps}: Cannot apply action cost: self.current_actions not found or empty.")

        else: # if episode_is_over
            logger.debug(f"CalcRewards Step {self.steps}: Episode is Over. Skipped shaping rewards.")

        # --- Log final reward components ---
        for agent in self.possible_agents:
             reward_component_log[agent]['final'] = rewards[agent]
        logger.debug(f"CalcRewards Step {self.steps}: Final Reward Components: {reward_component_log}")

        # --- Final check for NaN/Inf in rewards and replace with 0 ---
        for agent in self.possible_agents:
            if not np.isfinite(rewards[agent]):
                logger.error(f"!!! NaN or Inf detected in FINAL calculated reward for {agent} at step {self.steps}. Setting reward to 0. Components: {reward_component_log.get(agent, {})}")
                rewards[agent] = 0.0

        # --- Update previous distance for next step ---
        # Only update if the current distance was valid
        if current_docking_distance_finite:
            self.prev_docking_distance = docking_distance
            logger.debug(f"CalcRewards Step {self.steps}: Updated prev_docking_distance = {self.prev_docking_distance:.4f}")
        else:
            logger.warning(f"CalcRewards Step {self.steps}: Not updating prev_docking_distance due to current non-finite value ({docking_distance}). Kept {self.prev_docking_distance}")


        return rewards, terminations, truncations, infos


    def step(self, actions):
        """Steps the environment."""
        step_start_time = time.time()
        logger.debug(f"--- Step {self.steps} Called (Active Agents: {self.agents}) ---")

        # Ensure actions dict contains entries for currently active agents if provided
        self.current_actions = actions.copy() # Store actions for reward calc & logging
        active_actions = {}
        for agent in self.agents:
            if agent in actions:
                active_actions[agent] = actions[agent]
            else:
                logger.warning(f"Step {self.steps}: Action missing for active agent '{agent}'. Applying default zero action.")
                active_actions[agent] = np.zeros(self.action_space(agent).shape, dtype=np.float32)

        logger.debug(f"Step {self.steps}: Actions to apply: {active_actions}")

        # Apply actions to simulation
        try:
            self._apply_actions(active_actions)
            # Log applied forces after application
            # logger.debug(f"Step {self.steps}: xfrc_applied state (first 2 bodies): {self.data.xfrc_applied[:2,:]}")
        except Exception as e:
            logger.exception(f"CRITICAL ERROR applying actions at step {self.steps}: {e}")
            # If applying actions fails catastrophically, terminate the episode
            rewards = {agent: env_config.REWARD_COLLISION for agent in self.possible_agents} # Assign penalty
            terminations = {agent: True for agent in self.possible_agents}
            truncations = {agent: False for agent in self.possible_agents}
            infos = {agent: {'status': 'action_apply_error'} for agent in self.possible_agents}
            observations = {agent: self._get_obs(agent) for agent in self.possible_agents} # Get last valid obs if possible
            self.agents = [] # End episode
            logger.error(f"Step {self.steps}: Terminating episode due to action application error.")
            return observations, rewards, terminations, truncations, infos

        # Step the MuJoCo simulation
        try:
             mujoco.mj_step(self.model, self.data)
             self.steps += 1
             # Log key state variables AFTER mj_step
             servicer_qpos_adr = self.joint_qpos_adr[env_config.SERVICER_AGENT_ID]
             servicer_qvel_adr = self.joint_qvel_adr[env_config.SERVICER_AGENT_ID]
             target_qpos_adr = self.joint_qpos_adr[env_config.TARGET_AGENT_ID]
             servicer_pos = self.data.qpos[servicer_qpos_adr : servicer_qpos_adr+3]
             servicer_vel = self.data.qvel[servicer_qvel_adr : servicer_qvel_adr+3]
             target_pos = self.data.qpos[target_qpos_adr : target_qpos_adr+3]
             num_contacts = self.data.ncon
             logger.debug(f"Step {self.steps} (Post mj_step): Servicer Pos={servicer_pos}, Vel={servicer_vel}; Target Pos={target_pos}; Contacts={num_contacts}")
             if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                 logger.error(f"!!! NaN/Inf detected in MuJoCo qpos/qvel immediately after mj_step {self.steps} !!!")

        except mujoco.FatalError as e:
             logger.exception(f"MUJOCO FATAL ERROR during mj_step at step {self.steps}: {e}. Simulation unstable?")
             # Terminate episode on MuJoCo instability
             rewards = {agent: env_config.REWARD_COLLISION * 2 for agent in self.possible_agents} # Larger penalty
             terminations = {agent: True for agent in self.possible_agents}
             truncations = {agent: False for agent in self.possible_agents}
             infos = {agent: {'status': 'mujoco_fatal_error'} for agent in self.possible_agents}
             observations = {agent: self._get_obs(agent) for agent in self.possible_agents} # Try get last obs
             self.agents = [] # End episode
             logger.error(f"Step {self.steps}: Terminating episode due to MuJoCo fatal error.")
             return observations, rewards, terminations, truncations, infos
        except Exception as e:
             logger.exception(f"Unexpected error during MuJoCo mj_step at step {self.steps}: {e}")
             # Terminate cautiously on unexpected errors too
             rewards = {agent: env_config.REWARD_COLLISION for agent in self.possible_agents}
             terminations = {agent: True for agent in self.possible_agents}
             truncations = {agent: False for agent in self.possible_agents}
             infos = {agent: {'status': 'mj_step_error'} for agent in self.possible_agents}
             observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
             self.agents = [] # End episode
             logger.error(f"Step {self.steps}: Terminating episode due to unexpected mj_step error.")
             return observations, rewards, terminations, truncations, infos


        # Calculate rewards, dones, and infos based on the new state
        rewards, terminations, truncations, infos = self._calculate_rewards_and_done()

        # Update the list of active agents
        previous_agents = self.agents[:]
        self.agents = [agent for agent in self.agents if not (terminations.get(agent, False) or truncations.get(agent, False))]
        if len(self.agents) < len(previous_agents):
            logger.debug(f"Step {self.steps}: Agents terminated/truncated. Remaining: {self.agents}")

        # Get observations for the *next* state for all possible agents
        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}

        # Log final values being returned by step()
        logger.debug(f"Step {self.steps}: Returning Rewards: {rewards}")
        logger.debug(f"Step {self.steps}: Returning Terminations: {terminations}")
        logger.debug(f"Step {self.steps}: Returning Truncations: {truncations}")
        for agent, obs in observations.items():
            obs_finite = np.all(np.isfinite(obs))
            # logger.debug(f"Step {self.steps}: Returning Obs '{agent}' (finite: {obs_finite}): {obs[:4]}...") # Log truncated obs
            if not obs_finite:
                 logger.error(f"!!! NaN/Inf DETECTED IN FINAL STEP OBSERVATION for {agent} at step {self.steps} !!!")


        # Rendering logic
        if self.render_mode == "human": self.render()
        elif self.render_mode == "rgb_array": self._render_frame()

        # Clean up stored actions for this step if they exist
        if hasattr(self, 'current_actions'): del self.current_actions

        step_duration = time.time() - step_start_time
        logger.debug(f"--- Step {self.steps-1} Finished. Duration: {step_duration:.4f}s ---") # Log step duration

        # --- Check if episode ended this step ---
        if not self.agents:
             logger.info(f"Episode ended at step {self.steps-1}. Terminations={terminations}, Truncations={truncations}")


        return observations, rewards, terminations, truncations, infos


    def _render_frame(self):
        if self.render_mode != "rgb_array": return
        if self.renderer is None:
            try:
                # Ensure GL context is available if needed (usually handled by Renderer)
                logger.debug("Initializing MuJoCo Renderer for rgb_array.")
                self.renderer = mujoco.Renderer(self.model, height=env_config.RENDER_HEIGHT, width=env_config.RENDER_WIDTH)
                logger.info("MuJoCo Renderer initialized for rgb_array.")
            except Exception as e:
                 logger.error(f"Error initializing MuJoCo Renderer: {e}. Disabling rendering.")
                 self.render_mode = None # Disable rendering if init fails
                 return
        try:
            # Ensure data state is consistent before rendering
            mujoco.mj_forward(self.model, self.data) # Might be redundant if called in step, but safe
            self.renderer.update_scene(self.data, camera="fixed_side")
            pixels = self.renderer.render()
            if pixels is not None:
                self.render_frames.append(pixels)
                # logger.debug(f"Rendered frame {len(self.render_frames)} for rgb_array")
            else:
                logger.warning("MuJoCo renderer returned None for rgb_array frame.")
        except Exception as e:
            logger.error(f"Error during frame rendering: {e}")
            # Consider disabling rendering if errors persist
            # self.render_mode = None

    def render(self):
        # logger.debug(f"Render called with mode: {self.render_mode}")
        if self.render_mode == "human":
            if HAS_MEDIAPY:
                 if self.renderer is None:
                      try:
                           logger.debug("Initializing MuJoCo Renderer for human mode.")
                           self.renderer = mujoco.Renderer(self.model, height=env_config.RENDER_HEIGHT, width=env_config.RENDER_WIDTH)
                           logger.info("MuJoCo Renderer initialized for human mode.")
                      except Exception as e:
                           logger.error(f"Render init error (human): {e}. Cannot render.")
                           self.render_mode = None
                           return
                 try:
                      # Ensure data state is consistent before rendering
                      mujoco.mj_forward(self.model, self.data)
                      self.renderer.update_scene(self.data, camera="fixed_side")
                      pixels = self.renderer.render()
                      if pixels is not None:
                           # logger.debug("Showing image with mediapy.")
                           media.show_image(pixels)
                           # Add a small delay for visualization
                           time.sleep(1.0 / self.metadata["render_fps"])
                      else:
                           logger.warning("MuJoCo renderer returned None (human).")
                 except Exception as e:
                      logger.error(f"Human rendering error: {e}")
                      # self.render_mode = None # Optionally disable on error
            else:
                 # Warn only once about missing mediapy
                 if not hasattr(self, '_human_render_warned'):
                      logger.warning("Human rendering requires 'mediapy'. Install with 'pip install mediapy'. Cannot show real-time video.")
                      self._human_render_warned = True
            return None # Return None for human mode according to Gymnasium standard
        elif self.render_mode == "rgb_array":
            # Ensure a frame is rendered if needed
            self._render_frame()
            if self.render_frames:
                # logger.debug(f"Returning last frame ({len(self.render_frames)}) for rgb_array.")
                return self.render_frames[-1]
            else:
                logger.warning("Render called in rgb_array mode, but no frames available. Returning black frame.")
                # Return a black frame if rendering failed or hasn't happened
                return np.zeros((env_config.RENDER_HEIGHT, env_config.RENDER_WIDTH, 3), dtype=np.uint8)
        else:
            # logger.debug("Render mode is None or unsupported.")
            return None # Return None if no rendering mode is set

    def close(self):
        if self.renderer:
            try:
                logger.debug("Closing MuJoCo renderer.")
                self.renderer.close()
            except Exception as e:
                logger.error(f"Error closing renderer: {e}")
            self.renderer = None
        logger.info("SatelliteMARLEnv closed.")