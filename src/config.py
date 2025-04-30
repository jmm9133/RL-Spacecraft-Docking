import numpy as np
import os

# --- Simulation ---
XML_FILE_PATH            = os.path.join(os.path.dirname(__file__),
                                       'xml_references', 'satellites.xml')
TIMESTEP                 = 0.01
MAX_STEPS_PER_EPISODE    = 6000  # you can lower to 3000 for faster turnaround

POTENTIAL_MAX = 100.0

# --- Agents & Spaces ---
SERVICER_AGENT_ID        = "servicer"
TARGET_AGENT_ID          = "target"
POSSIBLE_AGENTS          = [SERVICER_AGENT_ID, TARGET_AGENT_ID]
OBS_DIM_PER_AGENT        = 17  # 3(rel_pos) + 3(rel_vel) + 4(quat) + 3(ang_vel) + 4(rel_quat)
ACTION_DIM_PER_AGENT     = 6   # 3 force + 3 torque

# --- Action Scaling ---
ACTION_FORCE_SCALING     = 12.3
ACTION_TORQUE_SCALING    = 8.1
ACTION_FORCE_SCALING_TARGET = 5.0
ACTION_TORQUE_SCALING_TARGET = 1.0

# --- Rewards: Terminal (all divided by 100 relative to before) ---
REWARD_DOCKING_SUCCESS   =  369.0   # was 9000
REWARD_COLLISION         =  -0.9   # was -90
REWARD_OUT_OF_BOUNDS     = -963.0   # was -1800

# --- PBRS Shaping (Φ) weights ---
POTENTIAL_WEIGHT_DISTANCE = 3.6  # try 10–200; higher → stronger pull
POTENTIAL_WEIGHT_VELOCITY =   0.5  # try 0.1–1.0
POTENTIAL_WEIGHT_ORIENT   =   1.0  # try 0.0–2.0

POTENTIAL_DISTANCE_EPSILON = 1e-3
PBRS_GATE_DISTANCE         = 0.5   # full penalty inside 0.5 m
PBRS_GATE_FACTOR_FAR       = 0.5  # was 1e-6; small but nonzero

# --- Incremental Docking Bonuses ---
INCREMENTAL_BONUS_ALIGN_NEAR  = 1.0
INCREMENTAL_BONUS_SLOW_NEAR   = 1.0
INCREMENTAL_BONUS_DIST_FACTOR = 1.5
INCREMENTAL_BONUS_VEL_FACTOR  = 1.5
INCREMENTAL_BONUS_ORIENT_FACTOR = 1.5

# --- Action Cost Penalty ---
REWARD_WEIGHT_ACTION_COST  = -0.001  # was -0.0001; after scaling this gives ~–1 per step at full thrust

# --- Training Gamma (must match your PPO config) ---
POTENTIAL_GAMMA            = 0.99

# --- Docking Thresholds ---
DOCKING_DISTANCE_THRESHOLD = 0.18
DOCKING_VELOCITY_THRESHOLD = 0.18
DOCKING_ORIENT_THRESHOLD   = 0.18

# --- Out-of-Bounds cutoff ---
OUT_OF_BOUNDS_DISTANCE     = 5.0

# --- Rendering ---
RENDER_WIDTH               = 640
RENDER_HEIGHT              = 480
RENDER_FPS                 = 30

# --- Initial Conditions ---
INITIAL_POS_OFFSET_TARGET  = np.array([2.0, 0.5, 0.0])
INITIAL_POS_RANGE_Servicer = [[0, 0, 0], [0, 0, 0]]
INITIAL_POS_RANGE_TARGET   = [[-1.5, -1.5, -0.5], [1.5, 1.5, 0.5]]
INITIAL_VEL_RANGE          = [[-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]]
INITIAL_ANG_VEL_RANGE      = [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]

# --- Competitive Mode (unused for now) ---
COMPETITIVE_MODE           = False
