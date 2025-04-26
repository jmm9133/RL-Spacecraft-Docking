# src/train_marl.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True
torch.set_float32_matmul_precision("high")

import os

# ─── only see GPUs 0 & 1 ───────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_GPUS_TO_USE = 1
# ─── disable libuv (Windows PyTorch has no libuv) ──
#os.environ["USE_LIBUV"] = "0"
# import torch.distributed as _dist
# _orig_init = _dist.init_process_group

# def _patched_init(backend, *args, **kwargs):
#     if backend == "nccl":
#         backend = "gloo"
#     return _orig_init(backend, *args, **kwargs)

# _dist.init_process_group = _patched_init

# # # ─── pin PyTorch rendezvous to localhost ──────────
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "29500"

import torch
import torch.distributed as dist


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
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
import pprint # Keep pprint for cleaner dictionary printing

# --- Import plotting library ---
import matplotlib.pyplot as plt
# -----------------------------

from .rllib_satellite_wrapper import RllibSatelliteEnv
from .satellite_marl_env import raw_env as satellite_pettingzoo_creator
from . import config as env_config

# --- Configuration ---
TRAIN_ITERATIONS = 100
CHECKPOINT_FREQ = 20
RESULTS_DIR = "output/ray_results"
LOG_DIR = "output/logs"
EVAL_EPISODES = 3 # Increase slightly for more stable eval score
EVAL_MAX_STEPS = env_config.MAX_STEPS_PER_EPISODE
# --- Add Plot configuration ---
PLOT_FILENAME = "training_progress.png"
# ----------------------------

# --- Setup Logging ---
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"training_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")

# Remove existing handlers if any exist
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    handler.close()

# Configure root logger
logging.basicConfig( level=logging.DEBUG,
    format="%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s] %(message)s",
    handlers=[ logging.FileHandler(log_file), logging.StreamHandler(sys.stdout) ] )

# Set levels for libraries to avoid excessive noise
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.rllib").setLevel(logging.WARNING)
logging.getLogger("mujoco").setLevel(logging.WARNING)

# Get logger for this script
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info(f"Logging setup complete. Log file: {log_file}")


# --- RLlib Environment Creator ---
def rllib_env_creator(config_dict):
    config_dict = config_dict or {}
    logger.debug(f"Creating RllibSatelliteEnv with config: {config_dict}")
    return RllibSatelliteEnv(config_dict)

register_env("satellite_marl", rllib_env_creator)
logger.info("Registered RLlib environment 'satellite_marl'")

# --- Helper Functions ---
# (Keep run_evaluation_video function exactly as it was)
def run_evaluation_video(algo: Algorithm, pettingzoo_env_creator_func, num_episodes=1, max_steps=1000):
    logger.info("\n--- Running Evaluation & Recording Video ---")
    results_dir_abs = os.path.abspath(RESULTS_DIR)
    iteration_str = f"iter_{algo.iteration}" if hasattr(algo, 'iteration') and algo.iteration is not None else "final"
    video_path = os.path.join(results_dir_abs, f"evaluation_video_{iteration_str}.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    frames = []
    all_episode_rewards = {agent: [] for agent in env_config.POSSIBLE_AGENTS}

    # Check the actual properties set in the config for API stack
    use_new_api_inference = (
        getattr(algo.config, "enable_env_runner_and_connector_v2", False) and
        getattr(algo.config, "enable_rl_module_and_learner", False)
    )
    logger.info(f"Evaluation using new API stack: {use_new_api_inference}")

    possible_agents = None # To store agent names

    try:
        logger.debug("Creating evaluation PettingZoo environment.")
        eval_env_pettingzoo = pettingzoo_env_creator_func(render_mode="rgb_array")
        possible_agents = eval_env_pettingzoo.possible_agents # Get agent names
        logger.debug(f"Evaluation env created. Possible agents: {possible_agents}")

        rl_modules = {}
        action_spaces = {}
        if possible_agents:
             action_spaces = {aid: eval_env_pettingzoo.action_space(aid) for aid in possible_agents}
             logger.debug("Retrieved action spaces for evaluation.")

        if use_new_api_inference:
            try:
                if possible_agents:
                    rl_modules = {}
                    for agent_id in possible_agents:
                        try:
                            rl_modules[agent_id] = algo.get_module(agent_id)
                        except ValueError:
                             logger.warning(f"Could not get RLModule for agent '{agent_id}' during evaluation.")
                        except Exception as mod_get_err: # Catch other potential errors getting module
                            logger.warning(f"Error getting RLModule for agent '{agent_id}': {mod_get_err}")

                    retrieved_modules = list(rl_modules.keys())
                    if retrieved_modules:
                        logger.info(f"Successfully retrieved RLModules for evaluation (Agents: {retrieved_modules}).")
                    else:
                        logger.warning("No RLModules were retrieved for evaluation.")

                else:
                    logger.error("Cannot get RLModules: possible_agents list is empty.")
                    eval_env_pettingzoo.close()
                    return
            except Exception as module_err:
                logger.exception(f"Could not get RLModules. Cannot run evaluation. Error: {module_err}")
                eval_env_pettingzoo.close()
                return
        else:
            logger.info("Using deprecated compute_single_action for evaluation.")

        # Evaluation Loop
        for episode in range(num_episodes):
            logger.info(f"Starting Evaluation Episode: {episode + 1}/{num_episodes}")
            episode_rewards_this_ep = {agent: 0.0 for agent in possible_agents} if possible_agents else {}
            try:
                obs, info = eval_env_pettingzoo.reset()
                if not obs: # Check if obs is empty or None
                     logger.error(f"Eval Ep {episode+1}: Environment reset returned empty/None observations. Stopping evaluation.")
                     break
                logger.debug(f"Eval Ep {episode+1}: Reset complete. Initial Obs keys: {list(obs.keys())}")
            except Exception as reset_err:
                logger.exception(f"Eval Ep {episode+1}: Failed to reset environment: {reset_err}")
                break # Stop evaluation if reset fails

            # Initial frame rendering
            try:
                frame = eval_env_pettingzoo.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    frames.append(frame.astype(np.uint8))
                    logger.debug(f"Eval Ep {episode+1}: Rendered initial frame.")
                else: logger.warning(f"Eval Ep {episode+1}: Initial render returned None or invalid frame.")
            except Exception as render_err:
                logger.warning(f"Eval Ep {episode+1}: Initial render failed: {render_err}")

            step = 0
            terminated = False
            truncated = False
            # Loop while agents exist and max steps not reached
            while eval_env_pettingzoo.agents and step < max_steps and not terminated and not truncated:
                current_active_agents = eval_env_pettingzoo.agents[:]
                actions = {}
                active_obs = {}
                valid_obs_found = False
                for agent_id in current_active_agents:
                    if agent_id in obs and obs[agent_id] is not None:
                        # Basic check for observation validity (e.g., shape, finite values)
                        agent_obs_data = obs[agent_id]
                        if isinstance(agent_obs_data, np.ndarray) and np.all(np.isfinite(agent_obs_data)):
                            active_obs[agent_id] = agent_obs_data
                            valid_obs_found = True
                        else:
                             logger.warning(f"Eval Ep {episode+1}, Step {step}: Invalid observation data for active agent '{agent_id}'. Type: {type(agent_obs_data)}, Data: {agent_obs_data}")
                    else:
                        logger.warning(f"Eval Ep {episode+1}, Step {step}: Observation missing or None for active agent '{agent_id}'.")

                if not valid_obs_found and current_active_agents:
                     logger.error(f"Eval Ep {episode+1}, Step {step}: No valid observations found for any active agents {current_active_agents}. Ending episode.")
                     break # Stop episode if no agent has a valid observation

                # Action Computation
                try:
                    if use_new_api_inference:
                        # --- New API Stack Action Computation ---
                        batch_tensor_obs = {}
                        for aid, o in active_obs.items():
                            np_obs = np.asarray(o, dtype=np.float32)
                            if not np.all(np.isfinite(np_obs)):
                                logger.warning(f"Eval Ep {episode+1}, Step {step}: NaN/Inf in observation for agent {aid} just before batching. Clamping.")
                                np_obs = np.nan_to_num(np_obs, nan=0.0, posinf=1e6, neginf=-1e6)
                            batch_tensor_obs[aid] = torch.from_numpy(np.expand_dims(np_obs, axis=0)).float()

                        if not batch_tensor_obs:
                            logger.warning(f"Eval Ep {episode+1}, Step {step}: No observations to batch. Cannot compute actions.")
                            break

                        forward_outs = {}
                        for agent_id, module in rl_modules.items():
                            if agent_id in batch_tensor_obs:
                                input_dict = {"obs": batch_tensor_obs[agent_id]}
                                try:
                                    with torch.no_grad():
                                         if hasattr(module, "forward_inference"):
                                             forward_outs[agent_id] = module.forward_inference(input_dict)
                                         else:
                                             forward_outs[agent_id] = module.forward_exploration(input_dict, explore=False)
                                except Exception as module_fwd_err:
                                    logger.error(f"Eval Ep {episode+1}, Step {step}: Error during module forward pass for {agent_id}: {module_fwd_err}")
                                    forward_outs[agent_id] = None

                        for agent_id in current_active_agents:
                            if agent_id not in active_obs: continue

                            if agent_id in forward_outs and forward_outs[agent_id] is not None:
                                action_output = forward_outs[agent_id].get("actions", forward_outs[agent_id].get("action"))
                                if action_output is not None and isinstance(action_output, torch.Tensor):
                                    action_np = action_output.cpu().numpy()
                                    if action_np.ndim > 1 and action_np.shape[0] == 1:
                                        actions[agent_id] = np.squeeze(action_np, axis=0)
                                    else:
                                        actions[agent_id] = action_np
                                else:
                                    dist_inputs = forward_outs[agent_id].get('action_dist_inputs')
                                    if dist_inputs is not None:
                                        logger.debug(f"Eval Ep {episode+1}, Step {step}: Using 'action_dist_inputs' for {agent_id}.")
                                        if dist_inputs.shape[-1] == 2 * env_config.ACTION_DIM_PER_AGENT:
                                            action_dist = torch.distributions.Normal(dist_inputs[..., :env_config.ACTION_DIM_PER_AGENT], torch.exp(dist_inputs[..., env_config.ACTION_DIM_PER_AGENT:]))
                                            sampled_action_tensor = action_dist.mean
                                            action_np = sampled_action_tensor.cpu().numpy()
                                            actions[agent_id] = np.squeeze(action_np, axis=0)
                                        else:
                                             logger.error(f"Eval Ep {episode+1}, Step {step}: Shape mismatch for action_dist_inputs for {agent_id}. Got shape {dist_inputs.shape}, expected last dim {2 * env_config.ACTION_DIM_PER_AGENT}. Using random action.")
                                             actions[agent_id] = action_spaces[agent_id].sample()
                                    else:
                                        logger.error(f"Eval Ep {episode+1}, Step {step}: No 'actions'/'action' or 'action_dist_inputs' found in module output for {agent_id}. Keys: {forward_outs[agent_id].keys()}. Using random action.")
                                        actions[agent_id] = action_spaces[agent_id].sample()
                            else:
                                logger.warning(f"Eval Ep {episode+1}, Step {step}: No forward output or failed forward pass for active agent {agent_id} with observation. Using random action.")
                                if agent_id in action_spaces:
                                     actions[agent_id] = action_spaces[agent_id].sample()
                                else: logger.error(f"Cannot sample random action for {agent_id}, action space missing.")


                    else: # --- Old API Stack Action Computation ---
                        for agent_id, agent_obs in active_obs.items():
                            try:
                                if not np.all(np.isfinite(agent_obs)):
                                    logger.warning(f"Eval Ep {episode+1}, Step {step}: NaN/Inf obs for {agent_id} before compute_single_action. Clamping.")
                                    agent_obs = np.nan_to_num(agent_obs, nan=0.0, posinf=1e6, neginf=-1e6)
                                actions[agent_id] = algo.compute_single_action(observation=agent_obs, policy_id=agent_id, explore=False)
                            except Exception as csa_err:
                                logger.error(f"Eval Ep {episode+1}, Step {step}: Error in compute_single_action for {agent_id}: {csa_err}. Using random action.")
                                if agent_id in action_spaces: actions[agent_id] = action_spaces[agent_id].sample()
                                else: logger.error(f"Cannot sample random action for {agent_id}, action space missing.")

                except Exception as action_comp_err:
                    logger.exception(f"Eval Ep {episode+1}, Step {step}: Unexpected error during action computation: {action_comp_err}")
                    break

                actions_to_step = {aid: act for aid, act in actions.items() if aid in eval_env_pettingzoo.agents}
                if not actions_to_step and eval_env_pettingzoo.agents:
                    logger.warning(f"Eval Ep {episode+1}, Step {step}: No actions computed/valid for active agents {eval_env_pettingzoo.agents}. Stepping with empty dict.")

                try:
                    next_obs, rewards, terminations_dict, truncations_dict, info = eval_env_pettingzoo.step(actions_to_step)
                    obs = next_obs

                    for r_agent, r_val in rewards.items():
                        if not np.isfinite(r_val):
                            logger.warning(f"Eval Ep {episode+1}, Step {step}: NaN/Inf detected in reward for agent {r_agent}: {r_val}. Setting to 0.")
                            rewards[r_agent] = 0.0

                    terminated = any(terminations_dict.values())
                    truncated = any(truncations_dict.values())

                except Exception as step_err:
                    logger.exception(f"Eval Ep {episode+1}, Step {step}: Error during environment step: {step_err}")
                    break

                try:
                    frame = eval_env_pettingzoo.render()
                    if frame is not None and isinstance(frame, np.ndarray): frames.append(frame.astype(np.uint8))
                    else: logger.debug(f"Eval Ep {episode+1}, Step {step}: Render returned None or invalid frame.")
                except Exception as render_err:
                    logger.warning(f"Eval Ep {episode+1}, Step {step}: Render failed: {render_err}")

                for aid, r in rewards.items():
                    if aid in episode_rewards_this_ep:
                        episode_rewards_this_ep[aid] += r

                step += 1

            logger.info(f"Evaluation Episode {episode + 1} finished after {step} steps. Final Rewards: {episode_rewards_this_ep}")
            for agent_id, reward in episode_rewards_this_ep.items():
                 if agent_id in all_episode_rewards: all_episode_rewards[agent_id].append(reward)

        eval_env_pettingzoo.close()
        logger.debug("Evaluation environment closed.")

    except Exception as eval_setup_err:
        logger.exception(f"Error during evaluation setup or run: {eval_setup_err}")
        if 'eval_env_pettingzoo' in locals() and hasattr(eval_env_pettingzoo, 'close'):
            try: eval_env_pettingzoo.close()
            except Exception as close_err: logger.error(f"Error closing eval env after failure: {close_err}")

    if possible_agents:
        avg_rewards = {}
        num_completed_episodes = 0
        if possible_agents and possible_agents[0] in all_episode_rewards:
             num_completed_episodes = len(all_episode_rewards[possible_agents[0]])

        if num_completed_episodes > 0:
            for agent in possible_agents:
                rewards_list = all_episode_rewards.get(agent, [])
                if len(rewards_list) == num_completed_episodes:
                    avg_rewards[agent] = np.mean(rewards_list)
                else:
                    logger.warning(f"Inconsistent number of rewards recorded for agent {agent} ({len(rewards_list)}) vs expected ({num_completed_episodes}). Skipping avg calc for this agent.")
                    avg_rewards[agent] = float('nan')
            logger.info(f"Average Evaluation Rewards over {num_completed_episodes} completed episodes: {avg_rewards}")
        else:
             logger.warning("No completed evaluation episodes recorded. Cannot calculate average rewards.")
    else:
        logger.warning("Could not calculate average evaluation rewards (no possible_agents found).")

    if frames:
        logger.info(f"Saving evaluation video ({len(frames)} frames) to: {video_path}")
        try:
            imageio.mimsave(video_path, frames, fps=env_config.RENDER_FPS, quality=8)
            logger.info("Evaluation video saved successfully.")
        except Exception as video_err: logger.error(f"Failed to save evaluation video: {video_err}")
    else: logger.warning("No frames recorded during evaluation, video will not be saved.")
# --- End of run_evaluation_video ---
#-----LOG REWARDS FUNCTION-----
def log_reward_info(env, step, rewards):
    """
    Log the reward information for the current step
    """
    # Get reward info from environment
    if hasattr(env, 'get_reward_info'):
        try:
            reward_info = env.get_reward_info()
            
            # Log state metrics
            metrics = reward_info.get("state_metrics", {})
            logger.info(f"Step {step} Metrics: " 
                        f"Distance={metrics.get('distance', 'N/A'):.4f}m, "
                        f"RelVel={metrics.get('relative_velocity', 'N/A'):.4f}m/s, "
                        f"OrientErr={metrics.get('orientation_error', 'N/A'):.4f}rad")
            
            # Log potential
            potential = reward_info.get("potential", {})
            potential_diff = potential.get('gamma', 0.95) * potential.get('current', 0) - potential.get('previous', 0)
            logger.info(f"Step {step} Potential: " 
                        f"Φ(s')={potential.get('current', 'N/A'):.4f}, "
                        f"Φ(s)={potential.get('previous', 'N/A'):.4f}, "
                        f"γΦ(s')-Φ(s)={potential_diff:.4f}")
            
            # Log rewards
            rewards_str = ", ".join([f"{a}={r:.4f}" for a, r in rewards.items()])
            logger.info(f"Step {step} Rewards: {rewards_str}")
            
        except Exception as e:
            logger.error(f"Error logging reward info: {e}")
    else:
        logger.debug(f"Environment does not have get_reward_info method")

# --- Main Training Script ---
if __name__ == "__main__":

    logger.info("--- Starting MARL Training Script ---")
    logger.info(f"Environment Config Module: {env_config.__name__}")
    logger.info(f"RL Algorithm: PPO")
    logger.info(f"Training Iterations: {TRAIN_ITERATIONS}")
    logger.info(f"Results Directory: {RESULTS_DIR}")
    logger.info(f"Max Steps per Episode: {env_config.MAX_STEPS_PER_EPISODE}")

    logger.warning("Reminder: Check '/tmp' or Ray's configured temp directory disk usage. Clean if necessary.")

    ray_init_success = False
    try:
        cpu_count = os.cpu_count() or 1
        logger.info(f"Detected {cpu_count} CPUs.")
        num_gpus = NUM_GPUS_TO_USE
        ray.init(num_gpus=NUM_GPUS_TO_USE, num_cpus=16,
             local_mode=False, # Ensure not in local mode
             logging_level=logging.WARNING,
             ignore_reinit_error=True) 
        #         runtime_env={
        #             "env_vars": {
        #                 "CUDA_VISIBLE_DEVICES": "0,1",
        #                 "USE_LIBUV": "0",
        #                 "MASTER_ADDR": "127.0.0.1",
        #                 "MASTER_PORT": "29500",
        #             }
        #         },
                # local_mode=False, logging_level=logging.WARNING, ignore_reinit_error=True)
        logger.info("Ray initialized successfully.")
        ray_init_success = True
    except Exception as ray_init_err:
         logger.exception(f"Ray initialization failed: {ray_init_err}")
         sys.exit(1)

    policies = None
    try:
        logger.info("Creating temporary environment to get action/observation spaces...")
        temp_env_rllib = rllib_env_creator({})
        policies = {
            aid: PolicySpec(
                observation_space=temp_env_rllib.observation_space[aid],
                action_space=temp_env_rllib.action_space[aid]
            )
            for aid in temp_env_rllib.possible_agents
        }
        temp_env_rllib.close()
        logger.info("Observation/Action spaces retrieved successfully:")
        for aid, spec in policies.items():
            logger.info(f"  Agent '{aid}': Obs={spec.observation_space}, Act={spec.action_space}")
    except Exception as e:
        logger.exception("Failed to create temp env or retrieve spaces. Exiting.")
        if ray_init_success: ray.shutdown()
        sys.exit(1)

    logger.info("Configuring RLlib PPO Algorithm...")
    RESULTS_DIR_ABS = os.path.abspath(RESULTS_DIR)
    os.makedirs(RESULTS_DIR_ABS, exist_ok=True)
    logger.info(f"Absolute results path: {RESULTS_DIR_ABS}")

    config = None
    algo = None
    try:
        num_workers = max(1, (cpu_count) - 2)
        logger.info(f"Using {num_workers} environment runners (workers).")

        #rollout_fragment_length_estimate = 4000
        rollout_fragment_length_estimate = 16000
        effective_train_batch_size = num_workers * rollout_fragment_length_estimate

        logger.info(f"Setting rollout_fragment_length: {rollout_fragment_length_estimate}")
        logger.info(f"Setting train_batch_size: {effective_train_batch_size} (= num_workers * rollout_fragment_length)")

        config = (
            PPOConfig()
            .environment("satellite_marl", env_config={})
            .learners(
                num_learners=NUM_GPUS_TO_USE,
                num_gpus_per_learner=1,
                num_cpus_per_learner=1,
            )
            # .env_runners(
            #     num_env_runners=32,
            #     num_envs_per_env_runner=2,
            #     num_gpus_per_env_runner=0,
            # )
            .learners(
                num_learners=num_gpus,       # 2 learners → 2 GPUs
                num_gpus_per_learner=1,
                num_cpus_per_learner=16,
                num_aggregator_actors_per_learner=2,
            )
            .framework(
                "torch",
                #torch_amp_learner=True,
                torch_compile_learner=True,
                torch_compile_learner_dynamo_backend="inductor",
            )
            .env_runners(
                num_env_runners=num_workers,
                rollout_fragment_length=200,
                observation_filter="MeanStdFilter",
                num_envs_per_env_runner=1,
            )
            .training(
                gamma=env_config.POTENTIAL_GAMMA,
                lambda_=0.95,
                lr=5e-5,
                train_batch_size=effective_train_batch_size,
                model={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                    "vf_share_layers": False,
                },
                optimizer={},
                num_epochs=2, # Renamed from num_sgd_iter
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.001,
                kl_coeff=0.2,
                kl_target=0.01,
                grad_clip=0.5,
                use_gae=True,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            )
            # .resources(
            #     num_gpus=2 if torch.cuda.is_available() else 0,
            # )
            .debugging(
                log_level="WARN",
                seed=np.random.randint(0, 10000),
            )
            .fault_tolerance(
                 restart_failed_env_runners=True
            )
            .evaluation(
                evaluation_interval=CHECKPOINT_FREQ,
                evaluation_duration=EVAL_EPISODES,
                evaluation_duration_unit="episodes",
                evaluation_num_env_runners=1,
                evaluation_parallel_to_training=True,
                evaluation_config = PPOConfig.overrides(
                     explore=False,
                     observation_filter="MeanStdFilter",
                )
            )
            .reporting(
                 # metrics_smoothing_episodes=10, # REMOVED THIS LINE - Caused TypeError
                 # Other reporting options can go here if needed
                 min_sample_timesteps_per_iteration=effective_train_batch_size # Ensure enough samples are collected
            )
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
        )
        logger.info("PPO configuration object created.")
        # --- Keep the verification block ---
        logger.info("--- Verifying Final Configuration Before Build ---")
        resolved_config_dict = config.to_dict()
        # Check the relevant keys in the resolved dict. The exact keys might vary slightly by RLlib version.
        logger.info(f"Resolved Config -> Resources -> num_gpus: {resolved_config_dict.get('resources', {}).get('num_gpus')}")
        logger.info(f"Resolved Config -> Framework: {resolved_config_dict.get('framework_str')}") # Or maybe just 'framework'
        logger.info(f"Resolved Config -> API Stack Learner Enabled: {resolved_config_dict.get('enable_rl_module_and_learner')}")
        # Check where the model config ended up:
        logger.info(f"Resolved Config -> Training -> Model: {resolved_config_dict.get('training', {}).get('model')}")
        logger.info(f"Resolved Config -> RL Module -> Model Config: {resolved_config_dict.get('rl_module_spec', {}).get('model_config_dict')}") # Check potential new location
        logger.info("Building Algorithm...")
        #algo = config.build()
        algo = config.build_algo() # New way
        logger.info("Algorithm built successfully.")

        policy_class_name = "Unavailable"
        try:
             first_policy_id = list(policies.keys())[0]
             if getattr(algo.config, "enable_rl_module_and_learner", False):
                 try:
                     module_instance = algo.get_module(first_policy_id)
                     policy_class_name = module_instance.__class__.__name__
                 except Exception as module_get_err:
                     logger.warning(f"Could not get module instance for '{first_policy_id}': {module_get_err}")
             else:
                 if algo.workers and algo.workers.local_worker():
                      policy_class_name = algo.workers.local_worker().get_policy(first_policy_id).__class__.__name__
             logger.info(f"Using Policy/Module Class: {policy_class_name}")
        except Exception as policy_log_err:
             logger.warning(f"Could not retrieve policy/module class name: {policy_log_err}")

    except Exception as e:
        logger.exception("Failed during RLlib configuration or algorithm build. Exiting.")
        if algo:
            try: algo.stop()
            except: pass
        if ray_init_success: ray.shutdown()
        sys.exit(1)

    # --- Data Storage for Plotting ---
    iterations_list = []
    timesteps_list = []
    train_rewards_mean = []
    train_episode_lengths_mean = []
    eval_rewards_mean = []
    # --------------------------------

    logger.info(f"\n--- Starting Training for {TRAIN_ITERATIONS} iterations ---")
    start_time = time.time()
    checkpoint_path = None
    last_checkpoint_iter = -1
    training_successful = False
    i = -1
    last_total_steps = 0

    try:
        for i in range(TRAIN_ITERATIONS):
            iter_start_time = time.time()
            logger.info(f"--- Starting Training Iteration {i+1}/{TRAIN_ITERATIONS} ---")

            result = algo.train()
            # --- I ADDED THIS TEMPORARILY ---
            # logger.info("--- Full Result Dictionary Structure (Iteration {}) ---".format(i+1))
            # logger.info(pprint.pformat(result, indent=2, width=120))
            # logger.info("--- End Full Result Dictionary ---")
            # ----------------------------
            logger.info(f"--- Finished Training Iteration {i+1}/{TRAIN_ITERATIONS} ---")
            iter_time = time.time() - iter_start_time

            # --- Extract and Log Key Performance Metrics ---
            timesteps_total = result.get("timesteps_total", result.get("num_env_steps_sampled_lifetime", 0))
            timesteps_this_iter = timesteps_total - last_total_steps
            last_total_steps = timesteps_total

            sampler_results = result.get("sampler_results", result.get("env_runners", {}))
            ep_reward_mean_sample = sampler_results.get("episode_reward_mean", float('nan'))
            ep_len_mean_sample = sampler_results.get("episode_len_mean", float('nan'))

            eval_metrics = result.get("evaluation", {})
            ep_reward_mean_eval = eval_metrics.get("episode_reward_mean", float('nan'))

            # --- Append data for plotting ---
            iterations_list.append(i + 1)
            timesteps_list.append(timesteps_total)
            train_rewards_mean.append(ep_reward_mean_sample)
            train_episode_lengths_mean.append(ep_len_mean_sample)
            eval_rewards_mean.append(ep_reward_mean_eval)
            # --------------------------------

            logger.info(f"Iter: {i+1}, Timesteps: {timesteps_total} (+{timesteps_this_iter})")
            logger.info(f"  Episode Reward Mean (Train): {ep_reward_mean_sample:.3f}")
            logger.info(f"  Episode Length Mean (Train): {ep_len_mean_sample:.1f}")
            if np.isfinite(ep_reward_mean_eval):
                 logger.info(f"  Episode Reward Mean (Eval):  {ep_reward_mean_eval:.3f}")
            logger.info(f"  Iteration Time: {iter_time:.2f}s")

             # --- Log detailed policy/learner stats ---
            # Corrected path based on pprint output
            logger.info(f"--- Training Stats for Iteration {i+1} ---")
            learner_stats_all_policies = result.get("learners", {}) # Get the 'learners' dict
            if learner_stats_all_policies:
                for policy_id in policies.keys(): # Iterate through expected policy IDs ('servicer', 'target')
                    policy_stats = learner_stats_all_policies.get(policy_id) # Get stats dict for this specific policy_id
                    if policy_stats and isinstance(policy_stats, dict): # Check if stats exist and is a dictionary
                        logger.info(f"\n--- {policy_id.upper()} POLICY/MODULE METRICS ---")
                        # Log common metrics, using .get() for safety
                        logger.info(f"Total Loss: {policy_stats.get('total_loss', 'N/A')}")
                        logger.info(f"Policy Loss: {policy_stats.get('policy_loss', 'N/A')}")
                        logger.info(f"Value Function Loss: {policy_stats.get('vf_loss', 'N/A')}")
                        logger.info(f"Entropy: {policy_stats.get('entropy', 'N/A')}")
                        logger.info(f"KL Divergence: {policy_stats.get('mean_kl_loss', policy_stats.get('policy_kl', 'N/A'))}")
                        logger.info(f"VF Explained Variance: {policy_stats.get('vf_explained_var', 'N/A')}")
                        # Find grad norm key (might vary slightly)
                        grad_norm_key = next((k for k in policy_stats if 'grad_norm' in k.lower() or 'gradients' in k.lower()), None)
                        logger.info(f"Gradient Norm: {policy_stats.get(grad_norm_key, 'N/A')}")
                        # Find LR key (might vary slightly)
                        lr_key = next((k for k in policy_stats if 'lr' in k.lower() or 'learning_rate' in k.lower()), None)
                        logger.info(f"Learning Rate: {policy_stats.get(lr_key, 'N/A')}")
                    else:
                        # Log if stats for a specific policy_id are missing or not a dict
                        logger.warning(f"No valid detailed stats found for policy '{policy_id}' under 'learners' key.")
                        # Optionally log the content if it's not None but not a dict
                        # if policy_stats is not None:
                        #    logger.debug(f"Content under 'learners' -> '{policy_id}': {policy_stats}")

                # Also log __all_modules__ stats if they exist separately (can contain aggregated info)
                all_modules_stats = learner_stats_all_policies.get("__all_modules__")
                if all_modules_stats and isinstance(all_modules_stats, dict):
                    logger.info(f"\n--- __ALL_MODULES__ AGGREGATED METRICS ---")
                    logger.info(f"Total Trainable Parameters: {all_modules_stats.get('num_trainable_parameters', 'N/A')}")
                    logger.info(f"Total Module Steps Trained: {all_modules_stats.get('num_module_steps_trained', 'N/A')}")
                    # Add other relevant aggregated stats found in __all_modules__ if needed

            else:
                logger.warning("Learner stats dictionary ('learners') not found in result.")
            # --- End corrected stats logging section ---


            # --- Check for training stagnation or failure ---
            # total_loss = learner_info.get("__all_modules__",{}).get("total_loss", 0.0)
            # if np.isnan(total_loss) and i > 0:
            #      logger.error(f"NaN detected in total loss at iteration {i+1}. Stopping training.")
            #      logger.error(f"Detailed Learner Info: {pprint.pformat(learner_info)}")
            #      break

            if i > 5 and timesteps_this_iter <= 0:
                 prev_total_ts = timesteps_list[-2] if len(timesteps_list) > 1 else 0
                 if timesteps_total <= prev_total_ts:
                     logger.error(f"Training stalled: Total timesteps ({timesteps_total}) did not increase from previous iteration ({prev_total_ts}). Stopping training.")
                     break

            # --- Checkpoint Saving ---
            if (i + 1) % CHECKPOINT_FREQ == 0:
                try:
                    logger.info(f"Attempting to save checkpoint at iteration {i+1}...")
                    checkpoint_result = algo.save(checkpoint_dir=RESULTS_DIR_ABS)
                    if checkpoint_result and checkpoint_result.checkpoint and checkpoint_result.checkpoint.path:
                         checkpoint_path = str(checkpoint_result.checkpoint.path)
                         last_checkpoint_iter = i
                         logger.info(f"Checkpoint saved successfully at: {checkpoint_path}")
                    else:
                         ckpt_path_alt = checkpoint_result.get("checkpoint_path") if isinstance(checkpoint_result, dict) else None
                         if ckpt_path_alt:
                              checkpoint_path = ckpt_path_alt
                              last_checkpoint_iter = i
                              logger.info(f"Checkpoint saved successfully (alt path): {checkpoint_path}")
                         else:
                              logger.error(f"Checkpoint saving reported success, but no valid path found in result: {checkpoint_result}")
                except Exception as save_err:
                    logger.exception(f"Failed to save checkpoint at iteration {i+1}: {save_err}")

        # End of training loop

        if i == TRAIN_ITERATIONS - 1:
             training_successful = True
             logger.info(f"Completed {TRAIN_ITERATIONS} training iterations.")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
    except Exception as train_err:
        logger.exception(f"Error during training loop at iteration {i+1}: {train_err}")
    finally:
        # --- Post-Training Actions ---
        total_training_time = time.time() - start_time
        logger.info(f"\n--- Training Loop Finished ---")
        logger.info(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
        logger.info(f"Last completed iteration: {i+1 if i>=0 else 'N/A'}")

        # --- Save Final Checkpoint ---
        if algo:
            if training_successful or (i > last_checkpoint_iter and i >= 0):
                 try:
                     logger.info("Attempting to save final checkpoint...")
                     final_checkpoint_result = algo.save(checkpoint_dir=RESULTS_DIR_ABS)
                     final_ckpt_path = None
                     if hasattr(final_checkpoint_result, 'checkpoint') and final_checkpoint_result.checkpoint and hasattr(final_checkpoint_result.checkpoint, 'path'):
                          final_ckpt_path = str(final_checkpoint_result.checkpoint.path)
                     elif isinstance(final_checkpoint_result, dict) and "checkpoint_path" in final_checkpoint_result:
                           final_ckpt_path = final_checkpoint_result["checkpoint_path"]
                     elif isinstance(final_checkpoint_result, str):
                          final_ckpt_path = final_checkpoint_result

                     if final_ckpt_path:
                          checkpoint_path = final_ckpt_path
                          logger.info(f"Final checkpoint saved successfully at: {checkpoint_path}")
                     else:
                           logger.error(f"Final checkpoint saving reported success, but no valid path found in result: {final_checkpoint_result}")
                 except Exception as final_save_err:
                     logger.exception(f"Failed to save final checkpoint: {final_save_err}")
            elif checkpoint_path:
                 logger.info(f"Using last successful checkpoint from iteration {last_checkpoint_iter + 1}: {checkpoint_path}")
            else:
                 logger.warning("No checkpoint was saved during training.")

            # --- Generate Plots ---
            if iterations_list:
                 plot_save_path = os.path.join(RESULTS_DIR_ABS, PLOT_FILENAME)
                 logger.info(f"Generating training progress plots to: {plot_save_path}")
                 try:
                      eval_data_exists = any(np.isfinite(val) for val in eval_rewards_mean if val is not None)
                      num_plots = 2 + (1 if eval_data_exists else 0)

                      fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
                      if num_plots == 1: axes = [axes]

                      plot_idx = 0

                      axes[plot_idx].plot(timesteps_list, train_rewards_mean, label="Episode Reward Mean (Train)", marker='.', linestyle='-', markersize=4, alpha=0.8)
                      axes[plot_idx].set_ylabel("Episode Reward Mean")
                      axes[plot_idx].set_title("Training Rewards")
                      axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
                      axes[plot_idx].legend()
                      plot_idx += 1

                      axes[plot_idx].plot(timesteps_list, train_episode_lengths_mean, label="Episode Length Mean (Train)", marker='.', linestyle='-', markersize=4, color='orange', alpha=0.8)
                      axes[plot_idx].set_ylabel("Episode Length")
                      axes[plot_idx].set_title("Episode Length")
                      axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
                      axes[plot_idx].legend()
                      plot_idx += 1

                      if eval_data_exists:
                            valid_eval_indices = [idx for idx, val in enumerate(eval_rewards_mean) if val is not None and np.isfinite(val)]
                            if valid_eval_indices:
                                eval_timesteps = [timesteps_list[idx] for idx in valid_eval_indices]
                                eval_rewards_plot = [eval_rewards_mean[idx] for idx in valid_eval_indices]
                                axes[plot_idx].plot(eval_timesteps, eval_rewards_plot, label="Episode Reward Mean (Eval)", marker='o', linestyle='--', markersize=5, color='green')
                                axes[plot_idx].set_ylabel("Episode Reward Mean")
                                axes[plot_idx].set_title("Evaluation Rewards")
                                axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
                                axes[plot_idx].legend()
                                plot_idx += 1
                            else:
                                 logger.info("No finite evaluation reward data found for plotting.")


                      axes[-1].set_xlabel("Total Timesteps")
                      fig.suptitle("Training Progress", fontsize=16)
                      fig.tight_layout(rect=[0, 0.03, 1, 0.97])

                      plt.savefig(plot_save_path)
                      logger.info("Plots saved successfully.")
                      plt.close(fig)
                 except Exception as plot_err:
                      logger.exception(f"Failed to generate plots: {plot_err}")
            else:
                 logger.warning("No data collected (iterations_list is empty), skipping plot generation.")
            # --------------------

            # --- Run Final Evaluation ---
            if (training_successful or i >= 0) and checkpoint_path:
                try:
                    logger.info(f"Restoring algorithm from checkpoint: {checkpoint_path}")
                    algo.restore(checkpoint_path)
                    logger.info("Algorithm restored. Running final evaluation...")
                    run_evaluation_video(algo, satellite_pettingzoo_creator, num_episodes=EVAL_EPISODES, max_steps=EVAL_MAX_STEPS)
                except Exception as eval_err:
                    logger.exception(f"Final evaluation failed: {eval_err}")
            else:
                logger.warning("Skipping final evaluation video (training incomplete or no checkpoint).")

            # --- Stop the Algorithm ---
            logger.info("Stopping RLlib Algorithm...")
            try:
                algo.stop()
                logger.info("Algorithm stopped.")
            except Exception as stop_err: logger.error(f"Error stopping algorithm: {stop_err}")
        else:
            logger.warning("Algorithm object not available, cannot save checkpoint, plot, or evaluate.")

        # --- Shutdown Ray ---
        if ray_init_success:
            logger.info("Shutting down Ray...")
            ray.shutdown()
            logger.info("Ray shut down.")

        logger.info("Script finished.")
        # Close logging handlers
        for handler in logging.root.handlers[:]:
            try:
                handler.flush()
                handler.close()
                logging.root.removeHandler(handler)
            except Exception as log_close_err: print(f"Error closing logging handler: {log_close_err}")