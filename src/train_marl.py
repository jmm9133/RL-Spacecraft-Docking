# (Keep imports and other setup as before)
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
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import AgentID
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.policy.sample_batch import SampleBatch
import pprint # Import pprint for cleaner dictionary printing

from .rllib_satellite_wrapper import RllibSatelliteEnv
from .satellite_marl_env import raw_env as satellite_pettingzoo_creator
from . import config as env_config

# --- Configuration ---
TRAIN_ITERATIONS = 36000 # Increase this later if learning starts
CHECKPOINT_FREQ = 20
RESULTS_DIR = "output/ray_results"
LOG_DIR = "output/logs"
EVAL_EPISODES = 9
EVAL_MAX_STEPS = env_config.MAX_STEPS_PER_EPISODE

# --- Setup Logging ---
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"training_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")

# Remove existing handlers if any exist
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    handler.close()

# Configure root logger
logging.basicConfig( level=logging.DEBUG, # <--- Keep LEVEL DEBUG for detailed logs
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    handlers=[ logging.FileHandler(log_file), logging.StreamHandler(sys.stdout) ] )

# Set levels for libraries to avoid excessive noise (optional)
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.rllib").setLevel(logging.INFO) # Keep RLlib INFO unless debugging RLlib internals
logging.getLogger("mujoco").setLevel(logging.WARNING) # Reduce MuJoCo INFO messages if desired

# Get logger for this script
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Ensure this script's logger is also DEBUG

logger.info(f"Logging setup complete. Log file: {log_file}")


# --- RLlib Environment Creator ---
def rllib_env_creator(config_dict):
    config_dict = config_dict or {}
    # Ensure render_mode is not passed if not needed, or set appropriately
    # config_dict["render_mode"] = None # Usually set by RLlib worker/evaluation config
    logger.debug(f"Creating RllibSatelliteEnv with config: {config_dict}")
    return RllibSatelliteEnv(config_dict)

register_env("satellite_marl", rllib_env_creator)
logger.info("Registered RLlib environment 'satellite_marl'")

# --- Helper Functions ---
def run_evaluation_video(algo: Algorithm, pettingzoo_env_creator_func, num_episodes=1, max_steps=1000):
    logger.info("\n--- Running Evaluation & Recording Video ---")
    results_dir_abs = os.path.abspath(RESULTS_DIR)
    # Use algo.iteration if available, otherwise a timestamp or default
    iteration_str = f"iter_{algo.iteration}" if hasattr(algo, 'iteration') else "final"
    video_path = os.path.join(results_dir_abs, f"evaluation_video_{iteration_str}.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    frames = []
    all_episode_rewards = {agent: [] for agent in env_config.POSSIBLE_AGENTS}

    # --- FIX: Correct check for new API stack ---
    # Check the actual properties set in the config
    use_new_api_inference = (
        getattr(algo.config, "enable_env_runner_and_connector_v2", False) and
        getattr(algo.config, "enable_rl_module_and_learner", False)
    )
    logger.info(f"Evaluation using new API stack: {use_new_api_inference}")
    # --- End FIX ---

    possible_agents = None # To store agent names

    try:
        logger.debug("Creating evaluation PettingZoo environment.")
        # Pass evaluation-specific config if needed, e.g., different render mode
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
                    # get_module might need error handling if a module doesn't exist
                    rl_modules = {}
                    for agent_id in possible_agents:
                        try:
                            rl_modules[agent_id] = algo.get_module(agent_id)
                        except ValueError: # Handle case where module might not exist for an agent
                             logger.warning(f"Could not get RLModule for agent '{agent_id}' during evaluation.")
                    logger.info(f"Successfully retrieved RLModules for evaluation (Agents: {list(rl_modules.keys())}).")
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

        # (Keep the rest of the evaluation loop logic as before)
        for episode in range(num_episodes):
            logger.info(f"Starting Evaluation Episode: {episode + 1}/{num_episodes}")
            episode_rewards_this_ep = {agent: 0.0 for agent in possible_agents} if possible_agents else {}
            try:
                obs, info = eval_env_pettingzoo.reset()
                logger.debug(f"Eval Ep {episode+1}: Reset complete. Initial Obs keys: {list(obs.keys())}")
            except Exception as reset_err:
                logger.exception(f"Eval Ep {episode+1}: Failed to reset environment: {reset_err}")
                break # Stop evaluation if reset fails

            # Initial frame rendering
            try:
                frame = eval_env_pettingzoo.render()
                if frame is not None:
                    frames.append(frame.astype(np.uint8))
                    logger.debug(f"Eval Ep {episode+1}: Rendered initial frame.")
                else: logger.warning(f"Eval Ep {episode+1}: Initial render returned None.")
            except Exception as render_err:
                logger.warning(f"Eval Ep {episode+1}: Initial render failed: {render_err}")

            step = 0
            while eval_env_pettingzoo.agents and step < max_steps:
                current_active_agents = eval_env_pettingzoo.agents[:]
                actions = {}
                active_obs = {}
                valid_obs_found = False
                for agent_id in current_active_agents:
                    if agent_id in obs and obs[agent_id] is not None:
                        active_obs[agent_id] = obs[agent_id]
                        valid_obs_found = True
                    else:
                        logger.warning(f"Eval Ep {episode+1}, Step {step}: Observation missing or None for active agent '{agent_id}'.")

                if not valid_obs_found:
                     logger.error(f"Eval Ep {episode+1}, Step {step}: No valid observations found for any active agents {current_active_agents}. Ending episode.")
                     break

                try:
                    if use_new_api_inference:
                        batch_obs = {}
                        for aid, o in active_obs.items():
                            np_obs = np.asarray(o, dtype=np.float32)
                            if not np.all(np.isfinite(np_obs)):
                                logger.warning(f"Eval Ep {episode+1}, Step {step}: NaN/Inf in observation for agent {aid} before batching. Clamping.")
                                np_obs = np.nan_to_num(np_obs, nan=0.0, posinf=1e6, neginf=-1e6)
                            batch_obs[aid] = np.expand_dims(np_obs, axis=0)

                        if batch_obs:
                            batch_tensor_obs = {aid: torch.from_numpy(ob).float() for aid, ob in batch_obs.items()}
                        else:
                             logger.warning(f"Eval Ep {episode+1}, Step {step}: No observations to batch. Cannot compute actions.")
                             break

                        forward_outs = {}
                        for agent_id, module in rl_modules.items():
                             if agent_id in batch_tensor_obs:
                                input_dict = {SampleBatch.OBS: batch_tensor_obs[agent_id]}
                                try:
                                    with torch.no_grad():
                                        forward_outs[agent_id] = module.forward_exploration(input_dict)
                                except Exception as module_fwd_err:
                                     logger.error(f"Eval Ep {episode+1}, Step {step}: Error during module forward pass for {agent_id}: {module_fwd_err}")
                                     forward_outs[agent_id] = None

                        for agent_id in current_active_agents:
                             if agent_id not in active_obs: continue
                             if agent_id in forward_outs and forward_outs[agent_id] is not None:
                                action_output = forward_outs[agent_id].get(SampleBatch.ACTIONS)
                                if action_output is not None:
                                     action_tensor = action_output
                                     if isinstance(action_tensor, torch.Tensor):
                                         action_np = action_tensor.cpu().numpy()
                                         if action_np.ndim > 1 and action_np.shape[0] == 1:
                                              actions[agent_id] = np.squeeze(action_np, axis=0)
                                         else:
                                              actions[agent_id] = action_np
                                     else:
                                         logger.error(f"Eval Ep {episode+1}, Step {step}: Action output for {agent_id} is not a Tensor. Type: {type(action_tensor)}. Using random action.")
                                         actions[agent_id] = action_spaces[agent_id].sample()
                                else:
                                     dist_inputs = forward_outs[agent_id].get('action_dist_inputs')
                                     if dist_inputs is not None:
                                          logger.debug(f"Eval Ep {episode+1}, Step {step}: Using 'action_dist_inputs' for {agent_id}.")
                                          action_dist = TorchDiagGaussian(dist_inputs, None)
                                          sampled_action_tensor = action_dist.deterministic_sample()
                                          action_np = sampled_action_tensor.cpu().numpy()
                                          actions[agent_id] = np.squeeze(action_np, axis=0)
                                     else:
                                          logger.error(f"Eval Ep {episode+1}, Step {step}: No '{SampleBatch.ACTIONS}' or 'action_dist_inputs' found in forward output for {agent_id}. Keys: {forward_outs[agent_id].keys()}. Using random action.")
                                          actions[agent_id] = action_spaces[agent_id].sample()
                             else:
                                 logger.warning(f"Eval Ep {episode+1}, Step {step}: No forward output or failed forward pass for active agent {agent_id} with observation. Using random action.")
                                 actions[agent_id] = action_spaces[agent_id].sample()

                    else: # Old API
                        for agent_id, agent_obs in active_obs.items():
                            try:
                                if not np.all(np.isfinite(agent_obs)):
                                    logger.warning(f"Eval Ep {episode+1}, Step {step}: NaN/Inf obs for {agent_id} before compute_single_action. Clamping.")
                                    agent_obs = np.nan_to_num(agent_obs, nan=0.0, posinf=1e6, neginf=-1e6)
                                actions[agent_id] = algo.compute_single_action(observation=agent_obs, policy_id=agent_id, explore=False)
                            except Exception as csa_err:
                                logger.error(f"Eval Ep {episode+1}, Step {step}: Error in compute_single_action for {agent_id}: {csa_err}. Using random action.")
                                actions[agent_id] = action_spaces[agent_id].sample()

                except Exception as action_comp_err:
                    logger.exception(f"Eval Ep {episode+1}, Step {step}: Unexpected error during action computation: {action_comp_err}")
                    break

                actions_to_step = {aid: act for aid, act in actions.items() if aid in eval_env_pettingzoo.agents}
                if not actions_to_step and eval_env_pettingzoo.agents:
                    logger.warning(f"Eval Ep {episode+1}, Step {step}: No actions computed for active agents {eval_env_pettingzoo.agents}. Stepping with empty dict.")

                try:
                    obs, rewards, terminations, truncations, info = eval_env_pettingzoo.step(actions_to_step)
                    if any(np.isnan(r) for r in rewards.values()):
                         logger.warning(f"Eval Ep {episode+1}, Step {step}: NaN detected in rewards from env.step: {rewards}")
                except Exception as step_err:
                    logger.exception(f"Eval Ep {episode+1}, Step {step}: Error during environment step: {step_err}")
                    break

                try:
                    frame = eval_env_pettingzoo.render()
                    if frame is not None: frames.append(frame.astype(np.uint8))
                except Exception as render_err:
                    logger.warning(f"Eval Ep {episode+1}, Step {step}: Render failed: {render_err}")

                for aid, r in rewards.items():
                    if aid in episode_rewards_this_ep:
                        if np.isfinite(r): episode_rewards_this_ep[aid] += r
                        else: logger.warning(f"Eval Ep {episode+1}, Step {step}: NaN/Inf reward {r} for agent {aid}. Not accumulating.")
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
            except: pass

    if possible_agents:
        avg_rewards = {}
        num_completed_episodes = 0
        for agent in possible_agents:
            rewards_list = all_episode_rewards.get(agent, [])
            if agent == possible_agents[0]: num_completed_episodes = len(rewards_list)
            if rewards_list: avg_rewards[agent] = np.mean(rewards_list)
            else: avg_rewards[agent] = float('nan')
        logger.info(f"Average Evaluation Rewards over {num_completed_episodes} completed episodes: {avg_rewards}")
    else:
        logger.warning("Could not calculate average evaluation rewards (no possible_agents found).")

    if frames:
        logger.info(f"Saving evaluation video ({len(frames)} frames) to: {video_path}")
        try:
            imageio.mimsave(video_path, frames, fps=env_config.RENDER_FPS, quality=8)
            logger.info("Evaluation video saved successfully.")
        except Exception as video_err: logger.error(f"Failed to save evaluation video: {video_err}")
    else: logger.warning("No frames recorded during evaluation, video will not be saved.")


# --- Main Training Script ---
if __name__ == "__main__":

    logger.info("--- Starting MARL Training Script ---")
    logger.info(f"Environment Config: {env_config.__name__}")
    logger.info(f"RLlib PPO Algorithm")
    logger.info(f"Training Iterations: {TRAIN_ITERATIONS}")
    logger.info(f"Results Directory: {RESULTS_DIR}")

    # --- REMINDER: Address Disk Space ---
    logger.warning("Reminder: Check '/tmp/ray' disk usage. Clean or configure Ray's temp directory if it's over 95% full.")
    # ------------------------------------

    ray_init_success = False
    try:
        cpu_count = os.cpu_count() or 1
        logger.info(f"Detected {cpu_count} CPUs.")
        # Example: Configure temp dir if needed
        # ray.init(num_cpus=cpu_count, local_mode=False, logging_level=logging.WARNING, ignore_reinit_error=True, _temp_dir="/path/with/more/space/ray_spill")
        ray.init(num_cpus=cpu_count, local_mode=False, logging_level=logging.WARNING, ignore_reinit_error=True)
        logger.info("Ray initialized successfully.")
        ray_init_success = True
    except Exception as ray_init_err:
         logger.exception(f"Ray initialization failed: {ray_init_err}")
         sys.exit(1)

    logger.info("Creating temporary environment to get action/observation spaces...")
    policies = None
    try:
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
        num_workers = max(1, (cpu_count or 4) - 2)
        logger.info(f"Using {num_workers} environment runners (workers).")
        rollout_fragment_length_estimate = 200

        # --- FIX: Set train_batch_size exactly equal to samples collected per iter ---
        effective_train_batch_size = num_workers * rollout_fragment_length_estimate
        # --- End FIX ---

        logger.info(f"Setting rollout_fragment_length: {rollout_fragment_length_estimate}")
        logger.info(f"Setting train_batch_size: {effective_train_batch_size} (= num_workers * rollout_fragment_length)")

        config = (
            PPOConfig()
            .environment("satellite_marl", env_config={})
            .framework("torch")
            .env_runners(
                num_env_runners=num_workers,
                rollout_fragment_length=rollout_fragment_length_estimate,
                observation_filter="MeanStdFilter",
                num_envs_per_env_runner=1,
                num_cpus_per_env_runner=1,
            )
            .training(
                gamma=0.99,
                # --- Try reducing LR and VF clipping ---
                #lr=5e-5,        # Reduced learning rate
                lr=[
                    [0, 5e-5],       # Start at iteration 0 with 5e-5 (or your current stable LR)
                    [1500, 1e-5],   # Linearly decay to 1e-5 by iteration 15000
                    [3600, 5e-7]    # Linearly decay to 5e-6 by iteration 36000 (adjust iters)
                ],
                #vf_clip_param=20.0, # Significantly reduced VF clipping
                # ---
                #kl_coeff=0.2,
                clip_param=0.1,
                entropy_coeff=0.0005,
                grad_clip=0.5,
                train_batch_size=effective_train_batch_size,
                num_epochs=8,
                model={
                    "fcnet_hiddens": [1024, 1024],
                    "fcnet_activation": "tanh",
                    "vf_share_layers": False,
                    }
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            )
            .resources(
                num_gpus=1,
                )
            .debugging(
                log_level="INFO",
                seed=369
                )
            .fault_tolerance(
                 restart_failed_env_runners=True
                 )
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=EVAL_EPISODES,
                evaluation_duration_unit="episodes",
                evaluation_num_env_runners=1,
                evaluation_parallel_to_training=True,
                evaluation_config = PPOConfig.overrides(
                     explore=False,
                     #observation_filter="MeanStdFilter",
                     observation_filter="NoFilter",
                     num_cpus_per_env_runner=1
                )
            )
            .api_stack(
                enable_rl_module_and_learner=True,
            )
        )
        logger.info("PPO configuration object created.")

        logger.info("Building Algorithm...")
        algo = config.build_algo() # Use the new method name
        logger.info("Algorithm built successfully.")

        # Log policy/module class name for verification
        policy_class_name = "Unavailable"
        try:
             first_policy_id = list(policies.keys())[0]
             # --- FIX: Correct check for new API ---
             if getattr(algo.config, "enable_rl_module_and_learner", False):
                 try:
                     module_instance = algo.get_module(first_policy_id)
                     policy_class_name = module_instance.__class__.__name__
                 except Exception as module_get_err:
                     logger.warning(f"Could not get module instance for '{first_policy_id}': {module_get_err}")
             # --- End FIX ---
             else: # Fallback for old API if needed
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

    logger.info(f"\n--- Starting Training for {TRAIN_ITERATIONS} iterations ---")
    results = []
    start_time = time.time()
    checkpoint_path = None
    last_checkpoint_iter = -1
    training_successful = False
    i = -1 # Initialize iteration counter
    last_total_steps = 0 # Track steps for Ts(iter) calculation

    try:
        for i in range(TRAIN_ITERATIONS):
            iter_start_time = time.time()
            logger.debug(f"--- Starting Training Iteration {i+1}/{TRAIN_ITERATIONS} ---")
            result = algo.train()
            logger.debug(f"--- Finished Training Iteration {i+1}/{TRAIN_ITERATIONS} ---")
            results.append(result)
            iter_time = time.time() - iter_start_time

            # --- Extract and Log Key Metrics ---
            eval_metrics = result.get("evaluation", {})
            episode_reward_mean_eval = eval_metrics.get("episode_reward_mean", float('nan'))
            sampler_results = result.get("sampler_results", result.get("env_runners", {}))
            episode_reward_mean_sample = sampler_results.get("episode_reward_mean", float('nan'))

            if np.isfinite(episode_reward_mean_eval):
                episode_reward_mean_log = episode_reward_mean_eval
                reward_source = "Eval"
            elif np.isfinite(episode_reward_mean_sample):
                episode_reward_mean_log = episode_reward_mean_sample
                reward_source = "Sample"
            else:
                episode_reward_mean_log = float('nan')
                reward_source = "None"

            # --- FIX: Calculate Ts(iter) correctly ---
            timesteps_total = result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0))
            timesteps_this_iter = timesteps_total - last_total_steps
            last_total_steps = timesteps_total
            # --- End FIX ---

            # Extract learner losses & gradient norms
            learner_info = result.get("info", {}).get("learner", {})
            servicer_loss, target_loss = float('nan'), float('nan')
            servicer_grad_norm, target_grad_norm = float('nan'), float('nan')

            all_module_stats = learner_info.get("__all_modules__", {})

            # Look for stats under the specific agent ID first, then under __all_modules__
            servicer_stats = learner_info.get(env_config.SERVICER_AGENT_ID, all_module_stats.get(env_config.SERVICER_AGENT_ID, {}))
            target_stats = learner_info.get(env_config.TARGET_AGENT_ID, all_module_stats.get(env_config.TARGET_AGENT_ID, {}))

            if servicer_stats:
                 servicer_loss = servicer_stats.get("total_loss", float('nan'))
                 servicer_grad_norm = servicer_stats.get("gradients_default_optimizer_global_norm", float('nan')) # Adjust key if optimizer name differs

            if target_stats:
                 target_loss = target_stats.get("total_loss", float('nan'))
                 target_grad_norm = target_stats.get("gradients_default_optimizer_global_norm", float('nan'))

            log_msg = (f"Iter: {i+1}/{TRAIN_ITERATIONS}, "
                       f"Ts(iter): {timesteps_this_iter}, Ts(total): {timesteps_total}, "
                       f"Reward ({reward_source}): {episode_reward_mean_log:.2f}, "
                       f"Loss(serv): {servicer_loss:.3f}, Loss(targ): {target_loss:.3f}, "
                       # Add gradient norms to summary
                       f"GradN(serv): {servicer_grad_norm:.3f}, GradN(targ): {target_grad_norm:.3f}, "
                       f"Time: {iter_time:.2f}s")
            logger.info(log_msg)

            if (i + 1) % 10 == 0 or timesteps_this_iter == 0 or np.isnan(servicer_loss):
                 try:
                      logger.debug(f"Full result dict at iter {i+1}:\n{pprint.pformat(result, indent=2, width=120)}")
                 except Exception as pp_err:
                      logger.error(f"Could not pretty-print result dict: {pp_err}")
                      logger.debug(f"Raw result dict at iter {i+1}: {result}")

            # --- FIX: Termination Check based on DETAILED learner info ---
            # Check if stats were found and if they are NaN
            servicer_loss_is_nan = np.isnan(servicer_loss) if servicer_stats else True # Treat missing stats as NaN for check
            target_loss_is_nan = np.isnan(target_loss) if target_stats else True

            if servicer_loss_is_nan and target_loss_is_nan and i > 0: # Allow first iter
                 logger.error(f"NaN detected in detailed losses for both agents at iteration {i+1}. Stopping training.")
                 logger.error(f"Detailed Learner Info: {learner_info}") # Log the source dict
                 break
            # --- End FIX ---

            if i > 5 and timesteps_this_iter == 0: # Check for zero steps processed
                 prev_total_ts = results[-2].get("num_env_steps_sampled_lifetime", results[-2].get("timesteps_total", 0)) if len(results) > 1 else 0
                 if timesteps_total <= prev_total_ts:
                     logger.error(f"Training stalled: Total timesteps did not increase between iteration {i} ({prev_total_ts}) and {i+1} ({timesteps_total}). Stopping training.")
                     break

            # Checkpoint saving
            if (i + 1) % CHECKPOINT_FREQ == 0:
                try:
                    logger.info(f"Attempting to save checkpoint at iteration {i+1}...")
                    checkpoint_result = algo.save()
                    if checkpoint_result.checkpoint and checkpoint_result.checkpoint.path:
                         checkpoint_path = str(checkpoint_result.checkpoint.path)
                         last_checkpoint_iter = i
                         logger.info(f"Checkpoint saved successfully at: {checkpoint_path}")
                    else:
                         logger.error(f"Checkpoint saving reported success, but no valid path found in result: {checkpoint_result}")
                except Exception as save_err:
                    logger.exception(f"Failed to save checkpoint at iteration {i+1}: {save_err}")

        if i == TRAIN_ITERATIONS - 1:
             training_successful = True
             logger.info(f"Completed {TRAIN_ITERATIONS} training iterations.")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
    except Exception as train_err:
        logger.exception(f"Error during training loop at iteration {i+1}: {train_err}")
    finally:
        total_training_time = time.time() - start_time
        logger.info(f"\n--- Training Loop Finished ---")
        logger.info(f"Total Training Time: {total_training_time:.2f} seconds")
        logger.info(f"Last completed iteration: {i+1 if i>=0 else 'N/A'}")

        if algo:
            if training_successful or (checkpoint_path is None) or (i > last_checkpoint_iter and i >= 0):
                 try:
                     logger.info("Attempting to save final checkpoint...")
                     final_checkpoint_result = algo.save()
                     if final_checkpoint_result.checkpoint and final_checkpoint_result.checkpoint.path:
                          checkpoint_path = str(final_checkpoint_result.checkpoint.path)
                          logger.info(f"Final checkpoint saved successfully at: {checkpoint_path}")
                     else:
                           logger.error(f"Final checkpoint saving reported success, but no valid path found in result: {final_checkpoint_result}")
                 except Exception as final_save_err:
                     logger.exception(f"Failed to save final checkpoint: {final_save_err}")
            elif checkpoint_path:
                 logger.info(f"Using last successful checkpoint from iteration {last_checkpoint_iter + 1}: {checkpoint_path}")
            else:
                 logger.warning("No checkpoint was saved during training.")

            if (training_successful or i >= 0 or checkpoint_path):
                try:
                    logger.info("Running final evaluation...")
                    run_evaluation_video(algo, satellite_pettingzoo_creator, num_episodes=EVAL_EPISODES, max_steps=EVAL_MAX_STEPS)
                except Exception as eval_err:
                    logger.exception(f"Final evaluation failed: {eval_err}")
            else:
                logger.warning("Skipping final evaluation video (no checkpoint saved or training did not run).")

            logger.info("Stopping RLlib Algorithm...")
            try:
                algo.stop()
                logger.info("Algorithm stopped.")
            except Exception as stop_err: logger.error(f"Error stopping algorithm: {stop_err}")
        else: logger.warning("Algorithm object not available, cannot save or evaluate.")

        if ray_init_success:
            logger.info("Shutting down Ray...")
            ray.shutdown()
            logger.info("Ray shut down.")

        logger.info("Script finished.")
        for handler in logging.root.handlers[:]:
            try:
                handler.close()
                logging.root.removeHandler(handler)
            except Exception as log_close_err: print(f"Error closing logging handler: {log_close_err}")