# src/config.py
import numpy as np
import os

# --- Simulation ---
XML_FILE_PATH = os.path.join(os.path.dirname(__file__), 'xml_references', 'satellites.xml')
TIMESTEP = 0.01
MAX_STEPS_PER_EPISODE = 10000

# --- Environment ---
SERVICER_AGENT_ID = "servicer"
TARGET_AGENT_ID = "target"
POSSIBLE_AGENTS = [SERVICER_AGENT_ID, TARGET_AGENT_ID]
OBS_DIM_PER_AGENT = 13
ACTION_DIM_PER_AGENT = 6
# --- FIX: Reduce Action Scaling Drastically ---
ACTION_FORCE_SCALING = 100.0 # Back to original, maybe even higher?
ACTION_TORQUE_SCALING = 0.5 # Back to original
# --- End FIX ---

# --- Rewards ---
REWARD_DOCKING_SUCCESS = 100.0
REWARD_COLLISION = -50.0
REWARD_OUT_OF_BOUNDS = -20.0

REWARD_WEIGHT_DISTANCE = -0.5       # Keep distance penalty
REWARD_WEIGHT_VELOCITY_MAG = -0.0001   # Keep velocity penalty (Optional)
# --- FIX: Add Small Action Cost ---
REWARD_WEIGHT_ACTION_COST = -0.0001   # Penalize large actions 
# --- End FIX ---

# Docking Thresholds
DOCKING_DISTANCE_THRESHOLD = 0.1
DOCKING_VELOCITY_THRESHOLD = 0.1

# --- Rendering ---
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_FPS = 30

# --- Initial Conditions ---
INITIAL_POS_RANGE = [[-1, -1, -1], [1, 1, 1]]
INITIAL_VEL_RANGE = [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]