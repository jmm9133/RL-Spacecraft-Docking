import os
import sys
import time
import datetime
import logging
import imageio.v2 as imageio
import numpy as np
import torch
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from pettingzoo import ParallelEnv
import mujoco
import gymnasium as gym
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("satellite_marl")

# Simple configuration
class Config:
    # Environment settings
    XML_FILE_PATH = "/Users/josephmurray/Documents/RL_Project/RL2/RL-Spacecraft-Docking/src/xml_references/satellites.xml"  # Put this file in the same directory
    MAX_STEPS_PER_EPISODE = 1000
    
    # Agent definitions
    SERVICER_AGENT_ID = "servicer"
    TARGET_AGENT_ID = "target"
    POSSIBLE_AGENTS = [SERVICER_AGENT_ID, TARGET_AGENT_ID]
    OBS_DIM_PER_AGENT = 13
    ACTION_DIM_PER_AGENT = 6  # 3 force + 3 torque
    
    # Action scaling
    ACTION_FORCE_SCALING = 5.0
    ACTION_TORQUE_SCALING = 0.1
    
    # Rewards
    REWARD_DOCKING_SUCCESS = 100.0
    REWARD_COLLISION = -50.0
    REWARD_WEIGHT_DISTANCE_DELTA = 5.0
    REWARD_WEIGHT_DISTANCE = -0.02
    REWARD_WEIGHT_VELOCITY_MAG = -0.1
    REWARD_WEIGHT_ACTION_COST = -0.000001
    
    # Docking thresholds
    DOCKING_DISTANCE_THRESHOLD = 0.1
    DOCKING_VELOCITY_THRESHOLD = 0.1
    VEL_PENALTY_EPSILON = 0.05
    
    # Rendering
    RENDER_WIDTH = 640
    RENDER_HEIGHT = 480
    RENDER_FPS = 30


class SatelliteMARLEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": Config.RENDER_FPS,
    }
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Load MuJoCo model
        xml_path = os.path.abspath(Config.XML_FILE_PATH)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.possible_agents = Config.POSSIBLE_AGENTS[:]
        self.render_mode = render_mode
        
        # Cache MuJoCo IDs
        self._get_mujoco_ids()
        
        # Initialize state variables
        self.agents = []
        self.steps = 0
        self.prev_docking_distance = float('inf')
        self.renderer = None
        self.render_frames = []
        self.current_actions = {}
    
    def _get_mujoco_ids(self):
        """Cache MuJoCo body, site, and joint IDs."""
        # Get body IDs
        self.body_ids = {
            Config.SERVICER_AGENT_ID: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, Config.SERVICER_AGENT_ID),
            Config.TARGET_AGENT_ID: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, Config.TARGET_AGENT_ID)
        }
        
        # Get site IDs
        self.site_ids = {
            "servicer_dock": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "servicer_dock_site"),
            "target_dock": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_dock_site")
        }
        
        # Get joint addresses
        servicer_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servicer_joint")
        target_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
        
        self.joint_qpos_adr = {
            Config.SERVICER_AGENT_ID: self.model.jnt_qposadr[servicer_jnt_id],
            Config.TARGET_AGENT_ID: self.model.jnt_qposadr[target_jnt_id]
        }
        self.joint_qvel_adr = {
            Config.SERVICER_AGENT_ID: self.model.jnt_dofadr[servicer_jnt_id],
            Config.TARGET_AGENT_ID: self.model.jnt_dofadr[target_jnt_id]
        }
    
    # PettingZoo API methods
    def observation_space(self, agent):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(Config.OBS_DIM_PER_AGENT,), dtype=np.float32)
    
    def action_space(self, agent):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(Config.ACTION_DIM_PER_AGENT,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial positions and orientations
        qpos_serv_start = self.joint_qpos_adr[Config.SERVICER_AGENT_ID]
        qvel_serv_start = self.joint_qvel_adr[Config.SERVICER_AGENT_ID]
        
        # Servicer starts at origin
        self.data.qpos[qpos_serv_start:qpos_serv_start+3] = [0, 0, 0]  # Position
        self.data.qpos[qpos_serv_start+3:qpos_serv_start+7] = [1, 0, 0, 0]  # Quaternion
        self.data.qvel[qvel_serv_start:qvel_serv_start+6] = [0, 0, 0, 0, 0, 0]  # Velocity
        
        # Target starts at a fixed position
        qpos_targ_start = self.joint_qpos_adr[Config.TARGET_AGENT_ID]
        qvel_targ_start = self.joint_qvel_adr[Config.TARGET_AGENT_ID]
        self.data.qpos[qpos_targ_start:qpos_targ_start+3] = [2.0, 0.5, 0.0]  # Position
        self.data.qpos[qpos_targ_start+3:qpos_targ_start+7] = [1, 0, 0, 0]  # Quaternion
        self.data.qvel[qvel_targ_start:qvel_targ_start+6] = [0, 0, 0, 0, 0, 0]  # Velocity
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
        
        # Reset episode variables
        self.agents = self.possible_agents[:]
        self.steps = 0
        self.render_frames = []
        self.current_actions = {}
        
        # Calculate initial docking distance
        servicer_dock_pos = self.data.site_xpos[self.site_ids["servicer_dock"]]
        target_dock_pos = self.data.site_xpos[self.site_ids["target_dock"]]
        self.prev_docking_distance = np.linalg.norm(servicer_dock_pos - target_dock_pos)
        
        # Get observations
        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        
        if self.render_mode == "human":
            self.render()
        
        return observations, infos
    
    def _get_obs(self, agent):
        """Get observation for a specific agent."""
        servicer_qpos_adr = self.joint_qpos_adr[Config.SERVICER_AGENT_ID]
        servicer_qvel_adr = self.joint_qvel_adr[Config.SERVICER_AGENT_ID]
        target_qpos_adr = self.joint_qpos_adr[Config.TARGET_AGENT_ID]
        target_qvel_adr = self.joint_qvel_adr[Config.TARGET_AGENT_ID]
        
        servicer_pos = self.data.qpos[servicer_qpos_adr:servicer_qpos_adr+3]
        servicer_quat = self.data.qpos[servicer_qpos_adr+3:servicer_qpos_adr+7]
        servicer_vel = self.data.qvel[servicer_qvel_adr:servicer_qvel_adr+3]
        servicer_ang_vel = self.data.qvel[servicer_qvel_adr+3:servicer_qvel_adr+6]
        
        target_pos = self.data.qpos[target_qpos_adr:target_qpos_adr+3]
        target_quat = self.data.qpos[target_qpos_adr+3:target_qpos_adr+7]
        target_vel = self.data.qvel[target_qvel_adr:target_qvel_adr+3]
        target_ang_vel = self.data.qvel[target_qvel_adr+3:target_qvel_adr+6]
        
        # Calculate relative positions and velocities
        relative_pos_world = target_pos - servicer_pos
        relative_vel_world = target_vel - servicer_vel
        
        if agent == Config.SERVICER_AGENT_ID:
            obs = np.concatenate([
                relative_pos_world, relative_vel_world, servicer_quat, servicer_ang_vel
            ])
        elif agent == Config.TARGET_AGENT_ID:
            obs = np.concatenate([
                -relative_pos_world, -relative_vel_world, target_quat, target_ang_vel
            ])
        else:
            raise ValueError(f"Unknown agent ID: {agent}")
        
        # Make sure observations are finite and correct shape
        obs = np.nan_to_num(obs)
        if obs.shape[0] != Config.OBS_DIM_PER_AGENT:
            diff = Config.OBS_DIM_PER_AGENT - obs.shape[0]
            if diff > 0:
                obs = np.pad(obs, (0, diff), 'constant')
            elif diff < 0:
                obs = obs[:Config.OBS_DIM_PER_AGENT]
                
        return obs.astype(np.float32)
    
    def _apply_actions(self, actions):
        """Apply actions to the simulation."""
        self.data.xfrc_applied *= 0.0  # Reset forces from previous step
        
        for agent, action in actions.items():
            if agent not in self.agents:
                continue
            
            # Ensure action is numpy array
            action = np.asarray(action, dtype=np.float64)
            
            body_id = self.body_ids[agent]
            body_row_index = body_id - 1  # MuJoCo body IDs are 1-based
            
            # Scale and apply actions
            force = action[:3] * Config.ACTION_FORCE_SCALING
            torque = action[3:] * Config.ACTION_TORQUE_SCALING
            force_torque_6d = np.concatenate([force, torque])
            
            # Check for NaN/Inf in actions
            if not np.all(np.isfinite(force_torque_6d)):
                force_torque_6d = np.zeros(6)
            
            # Apply the calculated forces
            self.data.xfrc_applied[body_row_index, :] = force_torque_6d
    
    def step(self, actions):
        """Step the environment with actions."""
        # Store actions for reward calculation
        self.current_actions = actions.copy()
        
        # Apply actions to simulation
        self._apply_actions(actions)
        
        # Step the MuJoCo simulation
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        # Calculate rewards, dones, and infos
        rewards, terminations, truncations, infos = self._calculate_rewards_and_done()
        
        # Update active agents
        previous_agents = self.agents[:]
        self.agents = [agent for agent in self.agents if not (terminations.get(agent, False) or truncations.get(agent, False))]
        
        # Get observations for all possible agents
        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
        
        # Rendering
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "rgb_array":
            self._render_frame()
        
        return observations, rewards, terminations, truncations, infos
    
    def _calculate_rewards_and_done(self):
        """Calculate rewards, terminations, truncations, and infos."""
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        
        # Calculate docking status
        is_docked = False
        docking_distance = float('inf')
        relative_velocity_mag = float('inf')
        
        try:
            # Get positions for docking sites
            servicer_dock_pos = self.data.site_xpos[self.site_ids["servicer_dock"]]
            target_dock_pos = self.data.site_xpos[self.site_ids["target_dock"]]
            docking_distance = np.linalg.norm(servicer_dock_pos - target_dock_pos)
            
            # Get velocities
            servicer_qvel_adr = self.joint_qvel_adr[Config.SERVICER_AGENT_ID]
            target_qvel_adr = self.joint_qvel_adr[Config.TARGET_AGENT_ID]
            servicer_lin_vel = self.data.qvel[servicer_qvel_adr:servicer_qvel_adr+3]
            target_lin_vel = self.data.qvel[target_qvel_adr:target_qvel_adr+3]
            relative_velocity_mag = np.linalg.norm(servicer_lin_vel - target_lin_vel)
            
            # Check if docked
            is_docked = (docking_distance < Config.DOCKING_DISTANCE_THRESHOLD and
                         relative_velocity_mag < Config.DOCKING_VELOCITY_THRESHOLD)
        except Exception as e:
            logger.error(f"Error calculating docking status: {e}")
        
        # Check for collisions
        is_collision = False
        try:
            servicer_body_id = self.body_ids[Config.SERVICER_AGENT_ID]
            target_body_id = self.body_ids[Config.TARGET_AGENT_ID]
            
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]
                
                is_target_servicer_collision = ((geom1_body == servicer_body_id and geom2_body == target_body_id) or
                                               (geom1_body == target_body_id and geom2_body == servicer_body_id))
                
                if is_target_servicer_collision and contact.dist < 0.001:
                    if not is_docked:
                        is_collision = True
                        break
        except Exception as e:
            logger.error(f"Error checking collisions: {e}")
        
        # Termination conditions
        if is_docked:
            logger.info(f"Docking successful at step {self.steps}!")
            for agent in self.possible_agents:
                terminations[agent] = True
                rewards[agent] += Config.REWARD_DOCKING_SUCCESS
                infos[agent]['status'] = 'docked'
        elif is_collision:
            logger.info(f"Collision detected at step {self.steps}!")
            for agent in self.possible_agents:
                terminations[agent] = True
                rewards[agent] += Config.REWARD_COLLISION
                infos[agent]['status'] = 'collision'
        
        # Truncation after max steps
        if self.steps >= Config.MAX_STEPS_PER_EPISODE:
            logger.info(f"Max steps reached, truncating episode at step {self.steps}.")
            for agent in self.possible_agents:
                if not terminations[agent]:
                    truncations[agent] = True
                    infos[agent]['status'] = 'max_steps'
        
        # Calculate rewards if episode is not over
        episode_is_over = any(terminations.values()) or any(truncations.values())
        
        if not episode_is_over:
            # Distance delta reward
            dist_delta = self.prev_docking_distance - docking_distance
            dist_delta_reward = Config.REWARD_WEIGHT_DISTANCE_DELTA * dist_delta
            rewards[Config.SERVICER_AGENT_ID] += dist_delta_reward
            
            # Distance penalty
            dist_penalty = docking_distance * Config.REWARD_WEIGHT_DISTANCE
            rewards[Config.SERVICER_AGENT_ID] += dist_penalty
            
            # Velocity penalty (distance-weighted)
            vel_penalty = Config.REWARD_WEIGHT_VELOCITY_MAG * (relative_velocity_mag / (docking_distance + Config.VEL_PENALTY_EPSILON))
            rewards[Config.SERVICER_AGENT_ID] += vel_penalty
            
            # Action cost penalty
            if hasattr(self, 'current_actions') and self.current_actions:
                for agent_id, action in self.current_actions.items():
                    if agent_id in self.agents:
                        action_np = np.asarray(action)
                        action_norm = np.linalg.norm(action_np)
                        action_magnitude_penalty = action_norm * Config.REWARD_WEIGHT_ACTION_COST
                        rewards[agent_id] += action_magnitude_penalty
        
        # Update previous distance for next step
        self.prev_docking_distance = docking_distance
        
        return rewards, terminations, truncations, infos
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def _render_frame(self):
        if self.render_mode not in ["rgb_array", "human"]:
            return
        
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.model, height=Config.RENDER_HEIGHT, width=Config.RENDER_WIDTH)
            except Exception as e:
                logger.error(f"Error initializing renderer: {e}")
                self.render_mode = None
                return
        
        try:
            # Update scene
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera="fixed")  # Adjust camera name as needed
            
            pixels = self.renderer.render()
            if pixels is not None:
                self.render_frames.append(pixels)
                return pixels
        except Exception as e:
            logger.error(f"Error rendering frame: {e}")
        
        return None
    
    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None


# RLlib environment wrapper
class RllibSatelliteEnv(gym.Env):
    """RLlib wrapper for the SatelliteMARLEnv."""
    
    def __init__(self, config=None):
        config = config or {}
        self.env = SatelliteMARLEnv(render_mode=config.get("render_mode", None))
        self.agents = self.env.possible_agents
        self.possible_agents = self.env.possible_agents
        
        # Define spaces
        self.observation_space = gym.spaces.Dict({
            agent: self.env.observation_space(agent) for agent in self.possible_agents
        })
        self.action_space = gym.spaces.Dict({
            agent: self.env.action_space(agent) for agent in self.possible_agents
        })
        
        # Initialize state
        self.observations = None
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
    
    def reset(self, *, seed=None, options=None):
        self.observations, _ = self.env.reset(seed=seed, options=options)
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.agents = self.env.agents
        return self.observations, {}
    
    def step(self, actions):
        self.observations, rewards, self.terminations, self.truncations, infos = self.env.step(actions)
        self.agents = self.env.agents
        
        # Check if episode is done
        done = all(self.terminations.values()) or all(self.truncations.values()) or not self.agents
        truncated = all(self.truncations.values())
        terminated = all(self.terminations.values())
        
        return self.observations, rewards, terminated, truncated, infos
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


# RLlib environment creator function
def rllib_env_creator(config_dict):
    return RllibSatelliteEnv(config_dict)


# Register the environment with RLlib
register_env("satellite_marl", rllib_env_creator)


# Helper function for evaluation and video recording
def run_evaluation_video(algo, num_episodes=1, max_steps=1000):
    """Run evaluation and record video."""
    print("--- Running Evaluation & Recording Video ---")
    video_path = "evaluation_video.mp4"
    frames = []
    all_episode_rewards = {agent: [] for agent in Config.POSSIBLE_AGENTS}
    
    try:
        # Create evaluation environment
        eval_env = RllibSatelliteEnv({"render_mode": "rgb_array"})
        
        for episode in range(num_episodes):
            print(f"Starting Evaluation Episode: {episode + 1}/{num_episodes}")
            episode_rewards = {agent: 0.0 for agent in Config.POSSIBLE_AGENTS}
            
            obs, _ = eval_env.reset()
            terminated, truncated = False, False
            step = 0
            
            while not (terminated or truncated) and step < max_steps:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    actions[agent_id] = algo.compute_single_action(observation=agent_obs, policy_id=agent_id, explore=False)
                
                obs, rewards, terminated, truncated, _ = eval_env.step(actions)
                
                # Record frame
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)
                
                # Record rewards
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                step += 1
            
            print(f"Evaluation Episode {episode + 1} finished after {step} steps. Rewards: {episode_rewards}")
            for agent, reward in episode_rewards.items():
                all_episode_rewards[agent].append(reward)
        
        eval_env.close()
        
        # Calculate average rewards
        avg_rewards = {agent: np.mean(rewards) for agent, rewards in all_episode_rewards.items()}
        print(f"Average Evaluation Rewards: {avg_rewards}")
        
        # Save video
        if frames:
            print(f"Saving evaluation video ({len(frames)} frames) to: {video_path}")
            imageio.mimsave(video_path, frames, fps=Config.RENDER_FPS)
            print("Video saved successfully.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    print("--- Starting MARL Training Script ---")
    
    # Initialize Ray
    ray.init(num_cpus=os.cpu_count() or 4, logging_level=logging.WARNING)
    
    # Create temporary environment to get action/observation spaces
    temp_env = rllib_env_creator({})
    policies = {
        agent: PolicySpec(
            observation_space=temp_env.observation_space[agent],
            action_space=temp_env.action_space[agent]
        )
        for agent in temp_env.possible_agents
    }
    temp_env.close()
    
    # Configure PPO
    config = (
        PPOConfig()
        .environment("satellite_marl", env_config={})
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=4000,
            num_sgd_iter=10,
            model={"fcnet_hiddens": [256, 256]},
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=2)
    )
    
    # Build algorithm
    algo = config.build()
    
    # Train for 100 iterations
    print("--- Starting Training ---")
    TRAIN_ITERATIONS = 100
    for i in range(TRAIN_ITERATIONS):
        result = algo.train()
        
        # Log progress
        avg_reward = result.get("episode_reward_mean", 0)
        print(f"Iteration {i+1}/{TRAIN_ITERATIONS}: Avg Reward = {avg_reward:.2f}")
        
        # Evaluate and save video every 10 iterations
        if (i + 1) % 10 == 0:
            algo.save("checkpoint")
            run_evaluation_video(algo, num_episodes=1)
    
    # Run final evaluation
    print("--- Running Final Evaluation ---")
    run_evaluation_video(algo, num_episodes=1, max_steps=Config.MAX_STEPS_PER_EPISODE)
    
    # Save final checkpoint
    algo.save("checkpoint_final")
    
    # Clean up
    algo.stop()
    ray.shutdown()