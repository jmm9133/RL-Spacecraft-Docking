# src/config.py
import numpy as np
import os

# --- Simulation ---
XML_FILE_PATH = os.path.join(os.path.dirname(__file__), 'xml_references', 'satellites.xml')
TIMESTEP = 0.01
# Give it enough time, but maybe not extreme initially. Adjust based on stable speed.
MAX_STEPS_PER_EPISODE = 5000

# --- Environment ---
SERVICER_AGENT_ID = "servicer"
TARGET_AGENT_ID = "target"
# IMPORTANT: Strongly recommend fixing target stationary by removing its freejoint in XML.
# If target agent remains, ensure its actions are zeroed in environment step.
POSSIBLE_AGENTS = [SERVICER_AGENT_ID, TARGET_AGENT_ID] # Keep both if MARL setup needs it
OBS_DIM_PER_AGENT = 13  # Matches 3(rel_pos)+3(rel_vel)+4(quat)+3(ang_vel)
ACTION_DIM_PER_AGENT = 6 # 3 force + 3 torque


# ** Phase 1: Get Close (Dominant Reward) **
# Strong incentive (+ve weight) to reduce distance step-by-step
REWARD_WEIGHT_DISTANCE_DELTA = 200.0  # *** TUNE THIS *** (Try 5.0, 10.0, 20.0)

# Optional: Small penalty (+ve weight) only for distance *above* the close threshold
REWARD_WEIGHT_DISTANCE_FAR = 0.005    # *** TUNE THIS *** (Try 0.0, 0.01, 0.05)

# ** Phase 2: Stabilize & Align (Penalties when close) **
# Define the distance threshold for activating phase 2 penalties
CLOSE_DIST_THRESHOLD = 0.9           # *** TUNE THIS *** (e.g., 0.3, 0.5, 0.7)

# Strong penalty (+ve weight) for velocity ONLY when distance < CLOSE_DIST_THRESHOLD
REWARD_WEIGHT_VELOCITY_CLOSE = 0.5   # *** TUNE THIS *** (e.g., 0.1, 0.5, 1.0)

# Strong penalty (+ve weight) for orientation error ONLY when distance < CLOSE_DIST_THRESHOLD
# Set weight to 0.0 if not using orientation or if calculation is uncertain
REWARD_WEIGHT_ORIENT_CLOSE = 1.0     # *** TUNE THIS *** (e.g., 0.0, 0.5, 1.0, 2.0)

# --- Action Cost (Always Active) ---
REWARD_WEIGHT_ACTION_COST = -0.00001 # Keep very small (negative weight)


# --- Action Scaling (Low Baseline) ---
ACTION_FORCE_SCALING = 5.0  # Low value for stability focus
ACTION_TORQUE_SCALING = .10 # Low value for stability focus

# --- Rewards (Base) ---
REWARD_DOCKING_SUCCESS = 300.0 # High reward for success
REWARD_COLLISION = -50.0    # Significant penalty for collision
REWARD_OUT_OF_BOUNDS = -20.0  # Penalty for going too far (if implemented)

# --- Shaping Rewards (Simplified Baseline) ---
# Moderate positive incentive for getting closer step-by-step
REWARD_WEIGHT_DISTANCE_DELTA = 5.0
# Gentle negative gradient pulling towards the target
REWARD_WEIGHT_DISTANCE = -0.02
# Moderate penalty for relative velocity (using SIMPLE LINEAR penalty in env code for now)
REWARD_WEIGHT_VELOCITY_MAG = -0.1
# Minimal penalty for action magnitude
REWARD_WEIGHT_ACTION_COST = -0.000001

# --- Shaping Rewards (Potential-Based Approach) ---

# Weights for the Potential Function Phi = -W_dist*dist - W_vel*vel^2 - W_orient*orient_err
# Positive weights mean these components make the potential *less* negative (better) when minimized
POTENTIAL_WEIGHT_DISTANCE = 8.0   # higher → stronger pull
POTENTIAL_WEIGHT_VELOCITY = 0.5  # enough to encourage braking
POTENTIAL_WEIGHT_ORIENT   = 0.2   # optional, set 0 if orientation unused

# Weight for explicit Orientation Penalty (applied every step, not via potential change)
# This directly penalizes misalignment angle (radians)
REWARD_WEIGHT_ORIENT = 0.9 # Tune (e.g., 0.1 to 2.0) - SET TO 0 IF NOT USING ORIENTATION

# Weight for Action Cost (keep small)
REWARD_WEIGHT_ACTION_COST = -0.000001

# --- Training Gamma (Needed for potential-based shaping) ---
GAMMA = 0.99 # MUST MATCH the gamma in your training config!

# --- Docking Thresholds ---
DOCKING_DISTANCE_THRESHOLD = 0.1
DOCKING_VELOCITY_THRESHOLD = 0.1
# Add if using orientation in docking check:
DOCKING_ORIENT_THRESHOLD = 0.1 # Max radians error (e.g., ~5.7 degrees)


# --- Docking Thresholds

# -- Proximity Bonus Config (DISABLED for baseline) --
PROXIMITY_BONUS_WEIGHT = 0.0 # Set to 0 to disable this reward term
PROXIMITY_BONUS_DIST_THRESHOLD = 0.3 # Irrelevant when weight is 0
PROXIMITY_BONUS_VEL_THRESHOLD = 0.05 # Irrelevant when weight is 0

# -- Velocity Penalty Denominator Epsilon --
# (Only needed if using distance-weighted velocity penalty in env code)
VEL_PENALTY_EPSILON = 0.05

# --- Docking Thresholds ---
DOCKING_DISTANCE_THRESHOLD = 0.1 # Target distance for successful dock
DOCKING_VELOCITY_THRESHOLD = 0.1 # Target relative velocity for successful dock

# --- Rendering ---
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_FPS = 30

# --- Initial Conditions ---
# TODO: Implement randomization in reset() if needed later for robustness
INITIAL_POS_RANGE = [[-1, -1, -1], [1, 1, 1]] # Relative range if implemented
INITIAL_VEL_RANGE = [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]

# --- Training Gamma (if needed by potential-based rewards) ---
POTENTIAL_GAMMA = 0.90 # Define if using potential-based shaping requiring gamma
