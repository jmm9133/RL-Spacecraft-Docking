# src/config.py
import numpy as np
import os

# --- Simulation ---
XML_FILE_PATH = os.path.join(os.path.dirname(__file__), 'xml_references', 'satellites.xml')
TIMESTEP = 0.01
MAX_STEPS_PER_EPISODE = 1500

# --- Environment ---
SERVICER_AGENT_ID = "servicer"
TARGET_AGENT_ID = "target"
POSSIBLE_AGENTS = [SERVICER_AGENT_ID, TARGET_AGENT_ID]
OBS_DIM_PER_AGENT = 13
ACTION_DIM_PER_AGENT = 6
# --- FIX: Reduce Action Scaling Drastically ---
ACTION_FORCE_SCALING = 10.0 # Back to original, maybe even higher?
ACTION_TORQUE_SCALING = 1.0 # Back to original
# --- End FIX ---

# --- Rewards ---
REWARD_DOCKING_SUCCESS = 300.0
REWARD_COLLISION = -50.0
REWARD_OUT_OF_BOUNDS = -20.0 # Make sure OOB check is implemented if used

REWARD_WEIGHT_DISTANCE = -0.005       # Keep distance penalty (or slightly reduce, e.g., -0.02)

# --- FIX: Adjust Shaping Rewards ---
REWARD_WEIGHT_VELOCITY_MAG = -0.001   # Reduced penalty (Optional)
REWARD_WEIGHT_ACTION_COST = -0.000001   # Significantly reduce action cost penalty
REWARD_WEIGHT_DISTANCE_DELTA = 3.0   # *** Significantly increase reward for getting closer ***
# --- End FIX ---
# Docking Thresholds
DOCKING_DISTANCE_THRESHOLD = 0.1
DOCKING_VELOCITY_THRESHOLD = 0.1
REWEIGHT_VELOCITY_MAG=-0.001
# --- Rendering ---
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_FPS = 30

# --- Initial Conditions ---
INITIAL_POS_RANGE = [[-1, -1, -1], [1, 1, 1]]
INITIAL_VEL_RANGE = [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]