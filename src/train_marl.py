# src/train_marl.py
# (Keep imports and logging setup as before)
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

from .rllib_satellite_wrapper import RllibSatelliteEnv
from .satellite_marl_env import raw_env as satellite_pettingzoo_creator
from . import config as env_config

# --- Configuration ---
TRAIN_ITERATIONS = 100 # Increase this later if learning starts
CHECKPOINT_FREQ = 20
RESULTS_DIR = "output/ray_results"
LOG_DIR = "output/logs"
EVAL_EPISODES = 5
EVAL_MAX_STEPS = env_config.MAX_STEPS_PER_EPISODE

# --- Setup Logging ---
# (Keep logging setup as before)
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"training_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    handler.close()
logging.basicConfig( level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    handlers=[ logging.FileHandler(log_file), logging.StreamHandler(sys.stdout) ] )
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.rllib").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

# --- RLlib Environment Creator ---
def rllib_env_creator(config_dict):
    config_dict = config_dict or {}
    return RllibSatelliteEnv(config_dict)
register_env("satellite_marl", rllib_env_creator)

# --- Helper Functions ---
# (Keep run_evaluation_video as in the previous corrected version)
def run_evaluation_video(algo: Algorithm, pettingzoo_env_creator_func, num_episodes=1, max_steps=1000):
    logger.info("\n--- Running Evaluation & Recording Video ---")
    results_dir_abs = os.path.abspath(RESULTS_DIR)
    video_path = os.path.join(results_dir_abs, "evaluation_video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    frames = []
    all_episode_rewards = []
    use_new_api_inference = algo.config.enable_rl_module_and_learner

    try:
        eval_env_pettingzoo = pettingzoo_env_creator_func(render_mode="rgb_array")

        rl_modules = {}
        action_spaces = {aid: eval_env_pettingzoo.action_space(aid) for aid in eval_env_pettingzoo.possible_agents}

        if use_new_api_inference:
            try:
                rl_modules = {agent_id: algo.get_module(agent_id) for agent_id in eval_env_pettingzoo.possible_agents}
                logger.info("Successfully retrieved RLModules for evaluation.")
            except Exception as module_err:
                logger.exception(f"Could not get RLModules. Cannot run evaluation. Error: {module_err}")
                eval_env_pettingzoo.close()
                return
        else: logger.info("Using deprecated compute_single_action for evaluation.")


        for episode in range(num_episodes):
            logger.info(f"Evaluation Episode: {episode + 1}/{num_episodes}")
            episode_rewards = {agent: 0.0 for agent in eval_env_pettingzoo.possible_agents}
            obs, info = eval_env_pettingzoo.reset()
            terminated = {agent: False for agent in eval_env_pettingzoo.possible_agents}
            truncated = {agent: False for agent in eval_env_pettingzoo.possible_agents}
            step = 0

            try:
                frame = eval_env_pettingzoo.render()
                if frame is not None: frames.append(frame.astype(np.uint8))
            except Exception as render_err: logger.warning(f"Initial render failed: {render_err}")

            while eval_env_pettingzoo.agents:
                if step >= max_steps:
                    logger.info("Eval max steps reached.")
                    break

                actions = {}
                active_agents = eval_env_pettingzoo.agents[:]
                active_obs = {agent_id: obs[agent_id] for agent_id in active_agents if agent_id in obs}

                if not active_obs:
                     logger.warning(f"No observations for active agents {active_agents}, skipping step {step}.")
                     break

                try:
                    if use_new_api_inference:
                        batch_obs = {aid: np.expand_dims(o, axis=0) for aid, o in active_obs.items()}
                        batch_tensor_obs = {aid: torch.from_numpy(ob).float() for aid, ob in batch_obs.items()}

                        forward_outs = {}
                        for agent_id, module in rl_modules.items():
                             if agent_id in batch_tensor_obs:
                                input_dict = {SampleBatch.OBS: batch_tensor_obs[agent_id]}
                                with torch.no_grad():
                                    forward_outs[agent_id] = module.forward_inference(input_dict)

                        for agent_id in active_agents:
                             if agent_id in forward_outs:
                                dist_inputs = forward_outs[agent_id].get('action_dist_inputs')
                                if dist_inputs is not None:
                                     action_dist = TorchDiagGaussian(dist_inputs, None)
                                     sampled_action_tensor = action_dist.deterministic_sample()
                                     action_np = sampled_action_tensor.cpu().numpy()
                                     actions[agent_id] = np.squeeze(action_np, axis=0)
                                     if actions[agent_id].shape != (env_config.ACTION_DIM_PER_AGENT,):
                                          logger.error(f"Sampled action shape {actions[agent_id].shape} != expected {(env_config.ACTION_DIM_PER_AGENT,)}")
                                          actions[agent_id] = action_spaces[agent_id].sample()
                                else:
                                     logger.error(f"No 'action_dist_inputs' for {agent_id}. Keys: {forward_outs[agent_id].keys()}")
                                     actions[agent_id] = action_spaces[agent_id].sample()
                             elif agent_id in active_obs:
                                 logger.warning(f"No forward output for {agent_id}. Random action.")
                                 actions[agent_id] = action_spaces[agent_id].sample()

                    else: # Old API
                        for agent_id, agent_obs in active_obs.items():
                            actions[agent_id] = algo.compute_single_action(observation=agent_obs, policy_id=agent_id, explore=False)

                except Exception as action_err:
                    logger.exception(f"Error computing action during evaluation: {action_err}")
                    break

                actions_to_step = {aid: act for aid, act in actions.items() if aid in eval_env_pettingzoo.agents}
                if not actions_to_step:
                    logger.warning(f"No actions to step for agents {eval_env_pettingzoo.agents}")
                    break

                try:
                    obs, rewards, terminations, truncations, info = eval_env_pettingzoo.step(actions_to_step)
                except Exception as step_err:
                    logger.exception(f"Error during env step in evaluation: {step_err}")
                    break

                terminated.update(terminations)
                truncated.update(truncations)

                try:
                    frame = eval_env_pettingzoo.render()
                    if frame is not None: frames.append(frame.astype(np.uint8))
                except Exception as render_err: logger.warning(f"Render failed step {step}: {render_err}")

                for aid, r in rewards.items():
                    if aid in episode_rewards: episode_rewards[aid] += r
                step += 1

            logger.info(f"Evaluation Episode {episode + 1} Rewards: {episode_rewards}")
            all_episode_rewards.append(episode_rewards)

        eval_env_pettingzoo.close()

    except Exception as eval_setup_err:
        logger.exception(f"Error setting up evaluation: {eval_setup_err}")

    if all_episode_rewards:
         if 'eval_env_pettingzoo' in locals() and hasattr(eval_env_pettingzoo, 'possible_agents'):
             avg_rewards = {agent: np.mean([ep_rew.get(agent, 0.0) for ep_rew in all_episode_rewards])
                            for agent in eval_env_pettingzoo.possible_agents}
             logger.info(f"Average Evaluation Rewards over {len(all_episode_rewards)} episodes: {avg_rewards}")
         else: logger.warning("Could not calculate average evaluation rewards.")

    if frames:
        logger.info(f"Saving evaluation video to: {video_path}")
        try:
            imageio.mimsave(video_path, frames, fps=env_config.RENDER_FPS, quality=8)
        except Exception as video_err: logger.error(f"Failed to save video: {video_err}")
    else: logger.warning("No frames recorded for evaluation video.")


# --- Main Training Script ---
if __name__ == "__main__":

    logger.info("Initializing Ray...")
    try:
        ray.init(num_cpus=os.cpu_count() or 1, local_mode=False,
                 logging_level=logging.WARNING, ignore_reinit_error=True)
    except Exception as ray_init_err:
         logger.exception(f"Ray initialization failed: {ray_init_err}")
         sys.exit(1)

    logger.info("Creating temporary environment to get action/observation spaces...")
    try:
        temp_env_rllib = rllib_env_creator({})
        policies = { aid: PolicySpec( observation_space=temp_env_rllib.observation_space[aid],
                                      action_space=temp_env_rllib.action_space[aid])
                     for aid in temp_env_rllib.possible_agents }
        temp_env_rllib.close()
        logger.info("Spaces retrieved.")
    except Exception as e:
        logger.exception("Failed to create temp env/retrieve spaces. Exiting.")
        ray.shutdown()
        sys.exit(1)

    logger.info("Configuring RLlib PPO Algorithm...")
    RESULTS_DIR_ABS = os.path.abspath(RESULTS_DIR)
    os.makedirs(RESULTS_DIR_ABS, exist_ok=True)

    try:
        num_workers = max(1, (os.cpu_count() or 4) - 1)
        logger.info(f"Using {num_workers} environment runners (workers).")

        # --- Ensure train_batch_size is adequate ---
        # rollout_fragment_length default is 'auto', which RLlib tries to determine.
        # A common manual setting is 200. Let's estimate a minimum train_batch_size.
        # If fragment length is 200, train_batch_size should be >= 200 * num_workers.
        min_train_batch = 200 * num_workers
        effective_train_batch_size = max(4000, min_train_batch)
        logger.info(f"Setting train_batch_size to {effective_train_batch_size} (min required estimate: {min_train_batch})")
        # ------------------------------------------

        config = (
            PPOConfig()
            .environment("satellite_marl", env_config={})
            .framework("torch")
            .env_runners(
                num_env_runners=num_workers,
                rollout_fragment_length='auto', # Keep 'auto' for now, but be aware of its interaction with train_batch_size
                observation_filter="MeanStdFilter",
            )
            .training(
                gamma=0.96,
                lr=1e-5,
                kl_coeff=0.2,
                clip_param=0.2,
                # --- FIX: Try reducing vf_clip_param ---
                vf_clip_param=5.0, # Reduced from 10.0
                # --------------------------------------
                entropy_coeff=0.02,
                grad_clip=0.5,
                # --- FIX: Use calculated train_batch_size ---
                train_batch_size=effective_train_batch_size,
                # ------------------------------------------
                num_epochs=10,
                model={"fcnet_hiddens": [256, 256]}
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            )
            .resources(num_gpus=0)
            # --- FIX: Set RLlib log level to INFO/DEBUG for more details ---
            .debugging(log_level="INFO") # Changed from WARN
            # -------------------------------------------------------------
            .fault_tolerance(restart_failed_env_runners=True)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=EVAL_EPISODES,
                evaluation_duration_unit="episodes",
                evaluation_num_env_runners=1,
                evaluation_parallel_to_training=True,
                evaluation_config=AlgorithmConfig.overrides(
                     explore=False,
                     observation_filter="MeanStdFilter"
                )
            )
        )
    except Exception as e:
        logger.exception("Failed during RLlib configuration. Exiting.")
        ray.shutdown()
        sys.exit(1)

    logger.info("Building Algorithm...")
    algo = None
    try:
        algo = config.build()
        # (Keep policy class name retrieval as before)
        policy_class_name = "Unavailable"
        try:
             if algo.config.enable_rl_module_and_learner:
                 first_module_id = list(policies.keys())[0]
                 module_instance = algo.get_module(first_module_id)
                 policy_class_name = module_instance.__class__.__name__
             elif algo.workers and algo.workers.local_worker():
                 policy_class_name = algo.workers.local_worker().get_policy().__class__.__name__
        except Exception: logger.warning("Could not retrieve policy/module class name.")
        logger.info(f"Algorithm Built. Using Policy/Module Class: {policy_class_name}")

    except Exception as e:
        logger.exception("Failed to build RLlib algorithm. Exiting.")
        if algo: algo.stop()
        ray.shutdown()
        sys.exit(1)

    logger.info(f"\n--- Starting Training for {TRAIN_ITERATIONS} iterations ---")
    results = []
    start_time = time.time()
    checkpoint_path = None
    last_checkpoint_iter = -1
    i = -1

    try:
        for i in range(TRAIN_ITERATIONS):
            iter_start_time = time.time()
            result = algo.train()
            results.append(result)
            iter_time = time.time() - iter_start_time

            eval_metrics = result.get("evaluation", {})
            episode_reward_mean = eval_metrics.get("episode_reward_mean", float('nan'))
            if np.isnan(episode_reward_mean):
                 sampler_results = result.get("sampler_results", {})
                 episode_reward_mean = sampler_results.get("episode_reward_mean", float('nan'))

            timesteps_this_iter = result.get("num_env_steps_trained", 0)
            timesteps_total = result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0))

            learner_info = result.get("info", {}).get("learner", {})
            servicer_loss = float('nan')
            target_loss = float('nan')
            if learner_info:
                if env_config.SERVICER_AGENT_ID in learner_info:
                    servicer_loss = learner_info[env_config.SERVICER_AGENT_ID].get("total_loss", float('nan'))
                if env_config.TARGET_AGENT_ID in learner_info:
                    target_loss = learner_info[env_config.TARGET_AGENT_ID].get("total_loss", float('nan'))

            logger.info(f"Iter: {i+1}/{TRAIN_ITERATIONS}, "
                        f"Ts(iter): {timesteps_this_iter}, Ts(total): {timesteps_total}, "
                        f"Reward (Eval/Sample): {episode_reward_mean:.2f}, "
                        f"Loss(serv): {servicer_loss:.3f}, Loss(targ): {target_loss:.3f}, "
                        f"Time: {iter_time:.2f}s")

            # --- Add DEBUG log for full result dict periodically ---
            if (i + 1) % 10 == 0: # Log every 10 iters
                 logger.debug(f"Full result dict at iter {i+1}: {result}")
            # ------------------------------------------------------

            if (i + 1) % CHECKPOINT_FREQ == 0:
                try:
                    checkpoint_dir_obj = algo.save(RESULTS_DIR_ABS)
                    checkpoint_path = getattr(checkpoint_dir_obj, "path", checkpoint_dir_obj)
                    last_checkpoint_iter = i
                    logger.info(f"Checkpoint saved in directory {checkpoint_path}")
                except Exception as save_err: logger.error(f"Save failed: {save_err}")

    except KeyboardInterrupt: logger.warning("Training interrupted.")
    except Exception as e: logger.exception(f"Error during training: {e}")
    finally:
        total_training_time = time.time() - start_time
        logger.info(f"\n--- Training Finished ---")
        logger.info(f"Total Training Time: {total_training_time:.2f} seconds")

        if algo:
            if checkpoint_path is None or (i > last_checkpoint_iter and i >= 0):
                 try:
                     final_checkpoint_obj = algo.save(RESULTS_DIR_ABS)
                     checkpoint_path = getattr(final_checkpoint_obj, "path", final_checkpoint_obj)
                     logger.info(f"Final checkpoint saved: {checkpoint_path}")
                 except Exception as e: logger.error(f"Final save failed: {e}")
            elif checkpoint_path: logger.info(f"Using last checkpoint: {checkpoint_path}")
            else: logger.warning("No checkpoint saved.")
        else: logger.warning("Algo unavailable, cannot save final checkpoint.")

    if checkpoint_path and algo:
        try:
            run_evaluation_video(algo, satellite_pettingzoo_creator, num_episodes=EVAL_EPISODES, max_steps=EVAL_MAX_STEPS)
        except Exception as e: logger.exception(f"Evaluation failed: {e}")
    else: logger.warning("Skipping evaluation video (no checkpoint/algo).")

    logger.info("Shutting down Ray...")
    if algo:
        try: algo.stop()
        except Exception as e: logger.error(f"Error stopping algo: {e}")
    ray.shutdown()
    logger.info("Script finished.")
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)