# src/config.py
# --- (Keep Simulation, Environment, Dimensions, Action Scaling, Rendering, Initial Conditions as before) ---
import numpy as np
import os

# --- Simulation ---
XML_FILE_PATH = os.path.join(os.path.dirname(__file__), 'xml_references', 'satellites.xml')
TIMESTEP = 0.01
MAX_STEPS_PER_EPISODE = 9000 

# --- Environment ---
SERVICER_AGENT_ID = "servicer"
TARGET_AGENT_ID = "target"
POSSIBLE_AGENTS = [SERVICER_AGENT_ID, TARGET_AGENT_ID]
OBS_DIM_PER_AGENT = 14  # 3(rel_pos)+3(rel_vel)+4(quat)+3(ang_vel)
ACTION_DIM_PER_AGENT = 6 # 3 force + 3 torque

# --- Action Scaling ---
# Can be adjusted later for competitive/asymmetric control
ACTION_FORCE_SCALING = 4.5
ACTION_TORQUE_SCALING = 1.5

# --- Rewards: Terminal ---
REWARD_DOCKING_SUCCESS = 9000.0 # Increased emphasis on success
REWARD_COLLISION = -1000.0    # Significant penalty for non-docking collision
REWARD_OUT_OF_BOUNDS = -5000.0 # Penalty for drifting too far apart
RENDEZVOUS_POINT = np.array([0.0, 0.0, 0.0])  # Fixed point in 3D space
RENDEZVOUS_DISTANCE_THRESHOLD = 0.1  # Distance threshold for successful rendezvous
RENDEZVOUS_VELOCITY_THRESHOLD = 0.05  # Maximum allowed velocity at rendezvous
RENDEZVOUS_RELATIVE_VELOCITY_THRESHOLD = 0.03  # Maximum relative velocity between agents
RENDEZVOUS_ORIENTATION_THRESHOLD = 0.2  # Maximum orientation error (radians) at rendezvous
# --- Rewards: Potential-Based Shaping (Positive Potential Φ) ---
# Goal: Φ should be HIGHER (more positive) when closer to the goal state.
# Φ(s) = Wd / (dist + ε) - Wv * ||v_rel|| - Wo * θ_err
# Shaping reward r_shape = γ * Φ(s') - Φ(s)

# Weight for Distance component (Higher Wd -> Stronger pull when close)
# Make Wd large enough to dominate when distance is small.
POTENTIAL_WEIGHT_DISTANCE = 20

# Weight for Velocity penalty (Higher Wv -> Stronger penalty for high relative speed)
# Needs to be negative in the formula, so Wv here is positive.
POTENTIAL_WEIGHT_VELOCITY =  .1 #0.5 # *** TUNE THIS (e.g., 0.5, 1.0, 2.0) ***

# Weight for Orientation penalty (Higher Wo -> Stronger penalty for misalignment)
# Set to 0 if orientation isn't critical or stable yet. Needs negative in formula.
POTENTIAL_WEIGHT_ORIENT = .5 #2.0 # *** TUNE THIS (e.g., 0.0, 1.0, 5.0, 10.0) ***

# Small constant added to distance in denominator to avoid division by zero
POTENTIAL_DISTANCE_EPSILON = 1e-3

# --- Rewards: Action Cost ---
# Penalty for control effort (applied every step)
REWARD_WEIGHT_ACTION_COST = -0.0001

# --- Training Gamma  ---
POTENTIAL_GAMMA = 0.99 

# --- Docking Thresholds ---
DOCKING_DISTANCE_THRESHOLD = 0.1  # Target distance for successful dock
DOCKING_VELOCITY_THRESHOLD = 0.1  # Target relative velocity for successful dock
DOCKING_ORIENT_THRESHOLD = 0.36 # Max radians error (~11.5 degrees) - tune this!

# --- OOB Threshold (Example) ---
OUT_OF_BOUNDS_DISTANCE = 5.0 # Distance beyond which OOB penalty applies

# --- Rendering ---
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_FPS = 30

# --- Initial Conditions (Will be randomized) ---
INITIAL_POS_OFFSET_TARGET = np.array([4.0, 0.5, 0.0]) # Default starting offset
INITIAL_POS_RANGE_Servicer = [[0.0, 0, 0], [0.0, 0, 0]]  #Servicer starts at origin
INITIAL_POS_RANGE_TARGET = [[2.0, 0, 0], [2.0, 0, 0]] # Random offset RELATIVE to servicer origin
INITIAL_VEL_RANGE = [[-0.00, -0.00, -0.00], [0.00, 0.00, 0.00]] # Small random initial velocities
INITIAL_ANG_VEL_RANGE = [[-0.0, -0.0, -0.0], [0.0, 0.0, 0.0]] # Small random initial angular velocities

# --- Competitive Mode (Future Use) ---
COMPETITIVE_MODE = False
POTENTIAL_WEIGHT_DISTANCE_TARGET = 50.0 # Target might also want distance info
POTENTIAL_WEIGHT_ORIENT_TARGET = 5.0 # Target might care about relative orientation
REWARD_WEIGHT_ACTION_COST_TARGET = -0.0001
ACTION_FORCE_SCALING_TARGET = 5.0
ACTION_TORQUE_SCALING_TARGET = 1.0
EVASION_BONUS = 100.0 # Reward for target if it evades until timeout