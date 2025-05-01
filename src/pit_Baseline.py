import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

# Add the parent directory to path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Setup logging
logging.basicConfig(level=logging.ERROR, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PID_Controller")

class PIDController:
    """Simple PID controller for the satellite docking task."""
    
    def __init__(self, kp_pos=5.0, ki_pos=0.01, kd_pos=8.0, 
                 kp_att=2.0, ki_att=0.01, kd_att=3.0):
        # Position control gains
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos
        
        # Attitude control gains
        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att
        
        # Error integration and previous errors
        self.pos_error_integral = np.zeros(3)
        self.att_error_integral = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)
        
        # Time step for derivative and integral calculations
        self.dt = 0.05  # Assuming 20Hz control rate
        
        logger.info(f"PID Controller initialized with gains: "
                   f"Pos(P={kp_pos}, I={ki_pos}, D={kd_pos}), "
                   f"Att(P={kp_att}, I={ki_att}, D={kd_att})")
    
    def reset(self):
        """Reset error integrals and previous errors"""
        self.pos_error_integral = np.zeros(3)
        self.att_error_integral = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)
        logger.debug("PID controller state reset")
    
    def compute_action(self, obs):
        """
        Compute control action from observation
        
        Args:
            obs: Observation from environment (17 dim for satellite env)
                [0:3] - relative position (target - servicer)
                [3:6] - relative velocity
                [6:10] - servicer quaternion
                [10:13] - servicer angular velocity
        
        Returns:
            action: 6D array [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        """
        # Extract components from observation
        rel_pos = obs[0:3]  # Target relative to servicer
        rel_vel = obs[3:6]  # Target vel relative to servicer
        quat = obs[6:10]    # Servicer quaternion
        ang_vel = obs[10:13]  # Servicer angular velocity
        
        # Compute position error (we want to move servicer toward target, so error is the relative position)
        pos_error = rel_pos
        
        # For attitude control, we want to orient the servicer to point toward the target
        # Extract direction to target (normalize relative position)
        target_direction = rel_pos / (np.linalg.norm(rel_pos) + 1e-8)
        
        # Convert quaternion to rotation matrix to get current z-axis direction
        # This is a simplified conversion for demonstration, assuming the docking port is along the z-axis
        w, x, y, z = quat
        # Simplified conversion from quaternion to z-axis direction
        current_z = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ])
        
        # Attitude error: cross product between current z-axis and target direction
        # This gives rotation axis to align current_z with target_direction
        att_error = np.cross(current_z, target_direction)
        
        # Compute PID terms for position
        pos_p_term = self.kp_pos * pos_error
        self.pos_error_integral += pos_error * self.dt
        pos_i_term = self.ki_pos * self.pos_error_integral
        pos_d_term = self.kd_pos * (pos_error - self.prev_pos_error) / self.dt
        self.prev_pos_error = pos_error.copy()
        
        # Add damping based on velocity to help with stability
        vel_damping = -1.0 * rel_vel
        
        # Compute position control forces
        position_forces = pos_p_term + pos_i_term + pos_d_term + vel_damping
        
        # Clamp position forces to reasonable values
        position_forces = np.clip(position_forces, -5.0, 5.0)
        
        # Compute PID terms for attitude
        att_p_term = self.kp_att * att_error
        self.att_error_integral += att_error * self.dt
        att_i_term = self.ki_att * self.att_error_integral
        att_d_term = self.kd_att * (att_error - self.prev_att_error) / self.dt
        self.prev_att_error = att_error.copy()
        
        # Add damping based on angular velocity
        ang_damping = -0.8 * ang_vel
        
        # Compute attitude control torques
        attitude_torques = att_p_term + att_i_term + att_d_term + ang_damping
        
        # Clamp attitude torques to reasonable values
        attitude_torques = np.clip(attitude_torques, -2.0, 2.0)
        
        # As we get closer, reduce control effort to prevent overshooting
        distance = np.linalg.norm(rel_pos)
        if distance < 1.0:
            position_forces *= 0.5
            attitude_torques *= 0.7
        
        # Combine into final action: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        action = np.concatenate([position_forces, attitude_torques])
        
        # Ensure action is within [-1, 1] bounds as expected by the environment
        action = np.clip(action, -1.0, 1.0)
        
        return action

def evaluate_pid_controller(env, controller, episodes=5, render=True):
    """
    Evaluate the PID controller on the satellite docking environment
    
    Args:
        env: Satellite environment
        controller: PID controller
        episodes: Number of episodes to run
        render: Whether to render the environment
        
    Returns:
        results: Dictionary with episode results
    """
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'step_rewards': [],
        'distances': [],
        'statuses': []
    }
    
    for episode in range(episodes):
        logger.info(f"Starting Episode {episode+1}/{episodes}")
        controller.reset()
        
        ep_step_rewards = []
        ep_distances = []
        
        # Reset the environment
        observations, infos = env.reset()
        
        # Render initial state if enabled
        if render:
            env.render()
        
        # Run episode
        terminated = {a: False for a in env.possible_agents}
        truncated = {a: False for a in env.possible_agents}
        terminated["__all__"] = False
        truncated["__all__"] = False
        
        step = 0
        total_rewards = {agent: 0.0 for agent in env.possible_agents}
        
        while not (terminated["__all__"] or truncated["__all__"]):
            actions = {}
            
            # Compute action for servicer agent using PID controller
            for agent in env.agents:
                if agent == 'servicer':
                    actions[agent] = controller.compute_action(observations[agent])
                else:
                    # For target agent, just apply zero action (floating in space)
                    actions[agent] = np.zeros(env.action_space(agent).shape)
            
            # Step the environment
            observations, rewards, terminated, truncated, infos = env.step(actions)
            
            # Track rewards
            for agent in env.possible_agents:
                if agent in rewards:
                    total_rewards[agent] += rewards[agent]
            
            # Log step reward (servicer agent)
            if 'servicer' in rewards:
                ep_step_rewards.append(rewards['servicer'])
                
            # Log distance if available
            try:
                # Try to get reward info to access state metrics
                reward_info = env.get_reward_info()
                distance = reward_info['state_metrics']['distance']
                ep_distances.append(distance)
                if step % 20 == 0:  # Log every 20 steps to reduce output volume
                    logger.info(f"Episode {episode+1}, Step {step}: "
                               f"Distance={distance:.4f}, "
                               f"Reward={rewards.get('servicer', 0):.4f}")
            except (AttributeError, KeyError) as e:
                logger.warning(f"Couldn't access distance: {e}")
            
            # Render if enabled - let the environment handle the rendering
            if render:
                env.render()
                
            step += 1
            
            # Safety termination in case of very long episodes
            if step >= 500:
                logger.warning(f"Episode {episode+1} reached max 500 steps, terminating.")
                break
        
        # Episode summary
        final_status = "unknown"
        for agent in env.possible_agents:
            if agent in infos and 'status' in infos[agent]:
                final_status = infos[agent]['status']
                break
                
        logger.info(f"Episode {episode+1} finished after {step} steps")
        logger.info(f"Final status: {final_status}")
        logger.info(f"Total rewards: {total_rewards}")
        
        # Store episode results
        results['episode_rewards'].append({k: v for k, v in total_rewards.items()})
        results['episode_lengths'].append(step)
        results['step_rewards'].append(ep_step_rewards)
        results['distances'].append(ep_distances)
        results['statuses'].append(final_status)
    
    return results

def plot_results(results):
    """Plot the results of the evaluation with enhanced visualizations"""
    episodes = len(results['episode_rewards'])
    
    # Create more comprehensive plots with multiple figures
    # Figure 1: Episode summary (rewards and lengths)
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Episode rewards
    plt.subplot(2, 1, 1)
    servicer_rewards = [r.get('servicer', 0) for r in results['episode_rewards']]
    target_rewards = [r.get('target', 0) for r in results['episode_rewards']]
    
    x = range(1, episodes+1)  # Episode numbers starting from 1
    width = 0.35
    plt.bar([i-width/2 for i in x], servicer_rewards, width, alpha=0.7, label='Servicer')
    plt.bar([i+width/2 for i in x], target_rewards, width, alpha=0.7, label='Target')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards by Agent')
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Episode lengths
    plt.subplot(2, 1, 2)
    plt.bar(x, results['episode_lengths'], color='green', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('pid_episode_summary.png')
    
    # Figure 2: Rewards and distances per timestep for each episode
    max_steps = max([len(rewards) for rewards in results['step_rewards']])
    
    # For each episode, create a detailed plot
    for ep in range(min(episodes, 3)):  # Plot first 3 episodes at most
        plt.figure(figsize=(14, 12))
        
        # Plot 1: Cumulative reward over time
        plt.subplot(3, 1, 1)
        rewards = results['step_rewards'][ep]
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards, 'b-', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Episode {ep+1}: Cumulative Reward Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Step rewards
        plt.subplot(3, 1, 2)
        plt.plot(rewards, 'g-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title(f'Episode {ep+1}: Reward at Each Step')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Distance over time
        plt.subplot(3, 1, 3)
        if ep < len(results['distances']) and results['distances'][ep]:
            distances = results['distances'][ep]
            plt.plot(distances, 'r-', linewidth=2)
            # Add threshold line (adjust based on your environment's docking threshold)
            plt.axhline(y=0.1, color='blue', linestyle='--', label='Docking Threshold')
            plt.xlabel('Step')
            plt.ylabel('Distance (m)')
            plt.title(f'Episode {ep+1}: Distance to Target Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'pid_episode_{ep+1}_details.png')
    
    # Figure 3: Combined performance visualization
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Success rate
    plt.subplot(2, 2, 1)
    status_counts = {}
    for status in results['statuses']:
        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1
    
    status_labels = list(status_counts.keys())
    status_values = [status_counts[s] for s in status_labels]
    
    plt.pie(status_values, labels=status_labels, autopct='%1.1f%%', 
            startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Episode Outcomes')
    
    # Plot 2: Average rewards comparison
    plt.subplot(2, 2, 2)
    agent_labels = ['Servicer', 'Target']
    avg_rewards = [np.mean(servicer_rewards), np.mean(target_rewards)]
    plt.bar(agent_labels, avg_rewards, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel('Average Reward')
    plt.title('Average Reward by Agent')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Reward distribution
    plt.subplot(2, 2, 3)
    plt.boxplot([servicer_rewards, target_rewards], labels=agent_labels)
    plt.ylabel('Reward')
    plt.title('Reward Distribution by Agent')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: Distance vs reward correlation (for first episode)
    plt.subplot(2, 2, 4)
    if results['distances'] and results['step_rewards'] and len(results['distances'][0]) > 0:
        # Get min length to prevent index errors
        min_len = min(len(results['distances'][0]), len(results['step_rewards'][0]))
        distances = results['distances'][0][:min_len]
        step_rewards = results['step_rewards'][0][:min_len]
        
        plt.scatter(distances, step_rewards, alpha=0.5)
        plt.xlabel('Distance (m)')
        plt.ylabel('Reward')
        plt.title('Distance vs. Reward (Episode 1)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pid_performance_summary.png')
    
    # Close all figures to free memory
    plt.close('all')
    
    logger.info("Results plots saved to:")
    logger.info("- pid_episode_summary.png")
    logger.info("- pid_episode_1_details.png (and others if multiple episodes)")
    logger.info("- pid_performance_summary.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of episodes: {episodes}")
    print(f"Average episode length: {np.mean(results['episode_lengths']):.2f} steps")
    print(f"Average servicer reward: {np.mean(servicer_rewards):.2f}")
    print(f"Average target reward: {np.mean(target_rewards):.2f}")
    print(f"Success rate: {results['statuses'].count('docked') / episodes:.2%}")
    print(f"Statuses: {[s for s in results['statuses']]}")

def main():
    """Main function to set up the environment and run evaluation"""
    try:
        # Now try to import the environment correctly
        try:
            # First try direct import (if script is in the same directory as the environment)
            from satellite_marl_env import raw_env
            logger.info("Successfully imported environment via direct import")
        except ImportError:
            try:
                # Try importing from src package
                from src.satellite_marl_env import raw_env
                logger.info("Successfully imported environment from src package")
            except ImportError:
                # If both fail, try a package-relative import if we're inside the src directory
                # This assumes the script is in the src directory
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "satellite_marl_env", 
                    os.path.join(current_dir, "satellite_marl_env.py")
                )
                if spec:
                    satellite_marl_env = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(satellite_marl_env)
                    raw_env = satellite_marl_env.raw_env
                    logger.info("Successfully imported environment via spec loader")
                else:
                    raise ImportError("Could not locate satellite_marl_env.py")
        
        # Create the environment with rendering enabled - can be 'human' or 'rgb_array'
        env = raw_env(render_mode="rgb_array")
        
        # Create PID controller
        controller = PIDController(
            kp_pos=3.0, ki_pos=0.01, kd_pos=5.0,  # Position control gains
            kp_att=1.5, ki_att=0.01, kd_att=2.0   # Attitude control gains
        )
        
        # Evaluate controller with rendering enabled
        results = evaluate_pid_controller(
            env, 
            controller, 
            episodes=5, 
            render=True
        )
        
        # Plot and print results
        plot_results(results)
        
        # Close environment
        env.close()
        
    except ImportError as e:
        logger.error(f"Error importing environment: {e}")
        logger.error("Could not import the environment. Please verify your project structure.")
        logger.error(f"Current directory: {current_dir}")
        logger.error(f"Parent directory: {parent_dir}")
        logger.error(f"sys.path: {sys.path}")
        logger.error("""
        You may need to modify this script to correctly import your environment.
        Possible solutions:
        1. Place this script in the same directory as satellite_marl_env.py
        2. Modify the import statements to match your project structure
        3. Set the PYTHONPATH environment variable to include your project's root directory
        """)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()