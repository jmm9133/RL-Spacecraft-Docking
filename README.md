# Autonomous On-Orbit Servicing Simulation with MARL

## Overview

This project develops a simulation framework for autonomous on-orbit servicing (OOS) tasks, specifically focusing on docking maneuvers between a servicing satellite and a target satellite. It utilizes Multi-Agent Reinforcement Learning (MARL) to train agents capable of performing these tasks in a physics-based environment simulated using MuJoCo. The target satellite can potentially be cooperative, adversarial, or neutral, requiring robust MARL strategies.

## Key Features

*   **Multi-Agent Reinforcement Learning (MARL):** Employs the PettingZoo API for defining the interaction between the servicer and target agents.
*   **Physics-Based Simulation:** Uses the MuJoCo physics engine to simulate orbital mechanics (simplified as rigid body dynamics in free space for this version), thruster forces, and collisions.
*   **Realistic Constraints (Conceptual):** The framework is designed to incorporate constraints like fuel (proxied by action cost) and sensor limitations (implicit in the observation space), although detailed modeling is part of future work.
*   **Modular Environment:** The satellite models (`xml_references/satellites.xml`) and environment logic (`src/satellite_marl_env.py`) are separated.
*   **RLlib Integration:** Leverages Ray RLlib for training MARL agents, specifically using Independent PPO (MAPPO) as a baseline.
*   **Configurable:** Key simulation and training parameters are centralized in `src/config.py`.
*   **Visualization & Evaluation:** Supports rendering simulation videos and includes scripts for training and evaluating learned policies.

## Project Structure

```
.
├── notebooks/                 # Dev space for notebooks and etc. ;)
├── xml_references/
│   └── satellites.xml         # MuJoCo XML model definition for satellites
├── src/
│   ├── __init__.py            # Makes src a package
│   ├── satellite_marl_env.py  # PettingZoo ParallelEnv definition
│   ├── config.py              # Simulation, environment, and training configuration
│   └── train_marl.py          # Main script for training and evaluation using RLlib
├── output/                    # Default directory for logs, checkpoints, videos
│   ├── ray_results/           # Directory for Ray RLlib training outputs
│   └── *.mp4                  # Saved evaluation videos
├── requirements.txt           # Python package dependencies
└── README.md                  # This file
```

## Technology Stack

*   **Python 3.8+**
*   **MuJoCo:** Physics Simulation
*   **PettingZoo:** MARL Environment API
*   **Gymnasium:** Core RL Environment API (PettingZoo dependency)
*   **Ray RLlib:** MARL Training Framework
*   **PyTorch:** Deep Learning Backend for RLlib (TensorFlow is an alternative)
*   **NumPy:** Numerical computations
*   **ImageIO:** Video generation

## Setup and Installation

1.  **Prerequisites:**
    *   Python (3.8 or newer recommended BUT lower than 3.13).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/jmm9133/RL-Spacecraft-Docking.git
    cd RL-Spacecraft-Docking
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install PyTorch. If you prefer TensorFlow, install `tensorflow` instead of `torch` and modify the `.framework("torch")` line in `src/train_marl.py` to `.framework("tf2")`)*

## Running the Code

1.  **Navigate to the project root directory.**

2.  **Run the Training Script:**
    ```bash
    python -m src.train_marl
    ```
    *   This command executes the `train_marl.py` script as a module, correctly handling relative imports within the `src` package.
    *   Training will begin using Ray RLlib and the configured PPO algorithm.
    *   Progress logs will be printed to the console.
    *   Checkpoints and detailed logs (including TensorBoard files) will be saved under `output/ray_results/`.

3.  **Monitor Training (Optional):**
    *   Open a new terminal, activate the virtual environment, and run TensorBoard:
        ```bash
        tensorboard --logdir output/ray_results
        ```
    *   Navigate to the URL provided (usually `http://localhost:6006/`) in your web browser to view training metrics like episode rewards, policy loss, etc.

4.  **Evaluation:**
    *   After training completes (or is interrupted), the script will automatically run an evaluation using the last saved checkpoint.
    *   It will simulate several episodes using the learned policies and save a video of one or more episodes to `output/evaluation_video.mp4`.

## How it Works

### 1. Simulation Environment (MuJoCo & XML)

*   The physical interaction between the satellites is modeled in `xml_references/satellites.xml`.
*   This XML file defines:
    *   **Bodies:** `servicer` and `target` satellites with associated geometries (shapes, sizes, visual properties).
    *   **Joints:** `<freejoint>` elements allow both satellites 6 degrees of freedom (translation + rotation) in space. Damping is added for stability.
    *   **Sites:** Specific points fixed to the bodies (`servicer_dock_site`, `target_dock_site`) used for calculating docking distance and alignment.
    *   **Visuals & Lighting:** Cameras and lighting are defined for rendering.
    *   **(Future)** Actuators: Thrusters could be defined here for more realistic control (currently using `xfrc_applied`).

### 2. MARL Environment (PettingZoo)

*   `src/satellite_marl_env.py` implements the `pettingzoo.ParallelEnv` interface, which is suitable for simultaneous actions by agents.
*   **Agents:** Defines two agents: `servicer` (Agent 1) and `target` (Agent 2).
*   **Observation Space:** Each agent receives its own observation, crucial for decentralized execution. The observation typically includes:
    *   Relative position to the other satellite's docking port.
    *   Relative linear velocity.
    *   Its own orientation (quaternion).
    *   Its own angular velocity.
*   **Action Space:** Each agent outputs an action vector representing desired forces and torques. Currently, these are applied directly to the satellite's center of mass using MuJoCo's `data.xfrc_applied` for simplicity. A value of `[-1, 1]` is scaled to a configured force/torque range.
*   **Reward Structure:**
    *   **Sparse Reward:** A large positive reward (`REWARD_DOCKING_SUCCESS`) is given to both agents upon successful docking (sites close, relative velocity low).
    *   **Penalties:** Negative rewards are given for collisions (`REWARD_COLLISION`) or going out of bounds (not implemented yet).
    *   **Reward Shaping (Optional):** The `config.py` includes parameters for optional reward shaping (commented out by default) to guide learning:
        *   Penalizing distance to the target docking site.
        *   Penalizing high relative velocity.
        *   Penalizing large actions (proxy for fuel consumption).
*   **Episode Termination:** An episode ends if:
    *   Successful docking occurs (`terminations = True`).
    *   A collision occurs (`terminations = True`).
    *   The maximum number of steps is reached (`truncations = True`).

### 3. MARL Training (Ray RLlib & PPO)

*   `src/train_marl.py` orchestrates the training process using Ray RLlib.
*   **RLlib:** A scalable library for various RL algorithms, including MARL.
*   **Independent PPO (MAPPO):** The chosen algorithm is Proximal Policy Optimization (PPO) applied independently to each agent.
    *   Each agent (`servicer`, `target`) has its own PPO policy network and optimizer.
    *   Each agent learns based on its *own* observations and the rewards it receives.
    *   There is no explicit communication or coordination mechanism built into the algorithm itself (beyond observing the effects of the other agent's actions on the environment state).
    *   This is often a strong baseline for MARL problems and simpler to implement than centralized critic methods (like MADDPG) or communication protocols.
*   **Configuration (`PPOConfig`):** RLlib uses a configuration object to set up the environment, algorithm hyperparameters (learning rate, batch sizes, network architecture), multi-agent settings (policy definitions, agent-to-policy mapping), and rollout/evaluation parameters.
*   **Training Loop:** RLlib handles the distributed data collection (rollouts) and policy updates. The `algo.train()` call performs one iteration.
*   **Checkpointing:** The training progress (model weights) is saved periodically, allowing training to be resumed or policies to be evaluated later.

## Configuration

*   The `src/config.py` file centralizes important parameters:
    *   Path to the MuJoCo XML file.
    *   Simulation timestep and max episode length.
    *   Agent IDs.
    *   Observation/Action space dimensions.
    *   Reward values and shaping weights.
    *   Docking thresholds.
    *   Rendering settings.
    *   (Future) Initial condition randomization ranges.

## Challenges Addressed

*   **MARL:** Handled using PettingZoo for the environment API and RLlib with Independent PPO for training separate policies.
*   **Physics-Based Environment:** Implemented using MuJoCo for realistic rigid body dynamics simulation.
*   **Sparse Rewards:** The primary reward is sparse (only upon docking). Optional reward shaping terms are available in `config.py` to potentially ease exploration, though they require careful tuning.

## Future Work / Improvements

*   **Realistic Actuators:** Replace `xfrc_applied` with MuJoCo thruster actuators for more realistic fuel consumption and control limitations.
*   **Fuel Modeling:** Explicitly track and limit fuel for each agent, incorporating it into the state and reward.
*   **Sensor Noise/Limitations:** Add noise to observations or limit the field of view to simulate real sensors.
*   **Advanced Scenarios:** Implement more complex scenarios like tumbling targets, debris avoidance, or refueling mechanics.
*   **Initial Condition Randomization:** Implement randomization of starting positions, velocities, and orientations in `satellite_marl_env.py`'s `reset()` method for more robust training.
*   **Reward Shaping Tuning:** Experiment extensively with reward shaping terms to potentially improve sample efficiency.
*   **Hyperparameter Optimization:** Tune RLlib PPO hyperparameters for better performance.
*   **Alternative MARL Algorithms:** Explore other RLlib algorithms suitable for MARL (e.g., MADDPG for continuous control with a centralized critic, or QMIX/VDN for discrete actions if applicable).
*   **Target Behavior:** Implement different target agent policies (cooperative, adversarial, fixed trajectory) to train the servicer against various scenarios.
