# src/config.py
import os

# --- Environment Core Configuration ---
POSSIBLE_AGENTS = ["servicer", "target"]
SERVICER_AGENT_ID = "servicer"
TARGET_AGENT_ID = "target"
OBS_DIM_PER_AGENT = 14  # 3 (rel_pos) + 3 (rel_vel) + 4 (rel_quat) + 3 (rel_ang_vel) + 1 (distance) + 3 (future expansion)
ACTION_DIM_PER_AGENT = 6  # 3 (force) + 3 (torque)

# --- Environment Dynamics ---
# MuJoCo XML file path
XML_FILE_PATH = os.path.join(os.path.dirname(__file__), 'xml_references', 'satellites.xml')
TIMESTEP = 0.01

# --- Initial Conditions (Reduced ranges for more stable learning) ---
INITIAL_POS_RANGE_Servicer = [-0.5, 0.5]  # Range for servicer initial position - now smaller
INITIAL_POS_RANGE_TARGET = [1.0, 3.0]     # Range for relative target position - more reasonable
INITIAL_POS_OFFSET_TARGET = [1.5, 0.0, 0.0]  # Default offset if random placement fails
INITIAL_VEL_RANGE = [-0.2, 0.2]           # Reduced initial velocity range
INITIAL_ANG_VEL_RANGE = [-0.1, 0.1]       # Reduced initial angular velocity range

# --- Control Parameters ---
# Scaling factors for control forces/torques (reduced for smoother control)
ACTION_FORCE_SCALING = 5.0                # Reduced from previous value for smoother control
ACTION_TORQUE_SCALING = 2.0               # Reduced from previous value for smoother control

# Agent-specific scalings (if needed)
ACTION_FORCE_SCALING_SERVICER = 5.0
ACTION_TORQUE_SCALING_SERVICER = 2.0
ACTION_FORCE_SCALING_TARGET = 1.0
ACTION_TORQUE_SCALING_TARGET = 0.5

# --- Termination Conditions ---
DOCKING_DISTANCE_THRESHOLD = 0.05         # 5cm - exact docking
DOCKING_VELOCITY_THRESHOLD = 0.05         # 5cm/s - gentle approach
DOCKING_ORIENT_THRESHOLD = 0.087          # ~5 degrees in radians
OUT_OF_BOUNDS_DISTANCE = 10.0             # Maximum allowed separation distance

# --- Episode Parameters ---
MAX_STEPS_PER_EPISODE = 5000              # Maximum episode length

# --- Reward Parameters ---
# Terminal rewards
REWARD_DOCKING_SUCCESS = 100.0             # Reduced from 1000+ to be more proportional
REWARD_COLLISION = -30.0                  # Reduced magnitude
REWARD_OUT_OF_BOUNDS = -20.0              # Reduced magnitude

# Action cost weight (control effort penalty)
REWARD_WEIGHT_ACTION_COST = -0.01          # Reduced from -1.0 for better balance

# Competitive/Collaborative Mode
COMPETITIVE_MODE = False                  # Set to True for competitive environment

# --- Potential-Based Reward Shaping Parameters ---
POTENTIAL_WEIGHT_DISTANCE = 5.0           # Distance component weight (greatly reduced)
POTENTIAL_WEIGHT_VELOCITY = 1.0           # Velocity component weight (reduced)
POTENTIAL_WEIGHT_ORIENT = 1.0             # Orientation component weight (reduced)
POTENTIAL_DISTANCE_EPSILON = 0.1          # Small epsilon for numerical stability
POTENTIAL_GAMMA = 0.99                    # Discount factor for PBRS - MATCH TO YOUR RL ALGORITHM

# --- Rendering Parameters ---
RENDER_HEIGHT = 480
RENDER_WIDTH = 640
RENDER_FPS = 30