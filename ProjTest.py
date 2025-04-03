import mujoco
import mujoco.viewer
import numpy as np
sat_xml = """<mujoco model="satellites">
  <compiler angle="degree" />
  <option timestep="0.01" gravity="0 0 0" />
  <default>
    <!-- Default geom properties -->
    <geom contype="1" conaffinity="1" friction="0.1" density="1000"/>
  </default>

  <worldbody>
    <!-- Servicing Satellite -->
    <body name="servicer" pos="0 0 0">
      <!-- Main body (e.g., sphere) -->
      <geom type="sphere" size="0.5" rgba="0.5 0.5 0.5 1"/>
      <!-- Docking port protrusion -->
      <geom name="servicer_dock_port" type="cylinder" pos="0 0 0.5" size="0.1 0.2" euler="0 0 0" rgba="1 0 0 1"/>
      <!-- Docking detection site -->
      <site name="servicer_dock_site" pos="0 0 0.5" size="0.15" rgba="0 1 0 1" />
    </body>

    <!-- Target Satellite -->
    <body name="target" pos="2 0 0">
      <!-- Main body (e.g., sphere) -->
      <geom type="sphere" size="0.5" rgba="0.5 0.5 0.5 1"/>
      <!-- Docking cavity (for receiving the docking port) -->
      <geom name="target_dock_cavity" type="cylinder" pos="0 0 -0.5" size="0.15 0.25" euler="0 0 0" rgba="0 0 1 1"/>
      <!-- Docking detection site -->
      <site name="target_dock_site" pos="0 0 -0.5" size="0.15" rgba="0 1 0 1" />
    </body>
  </worldbody>
</mujoco>
"""
# Load the model
model = mujoco.MjModel.from_xml_string(sat_xml)

# Create simulation data
data = mujoco.MjData(model)

duration = 5.0  # seconds
dt = model.opt.timestep
steps = int(duration / dt)


sat_positions = np.zeros((steps, 3))
times = np.arange(0, duration, dt)
data.qpos[0:3] = np.array([0, 0, 2])  # x, y, z

# Velocity (x, y, z)
data.qvel[0:3] = np.array([0.5, 0, -1])
# Run simulations
for i in range(steps):
    # Store current positions
    sat_positions[i] = data.qpos[0:3]

    # Step simulation
    mujoco.mj_step(model, data)
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ----- Setup and Simulation -----
os.makedirs("output", exist_ok=True)

# Load the satellites XML model from the string
model = mujoco.MjModel.from_xml_string(satellites_xml)
data = mujoco.MjData(model)

# Set initial conditions for servicer and target
data.qpos[0:3] = np.array([0, 0, 0])      # Servicer position
data.qvel[0:3] = np.array([1, 0, 0])      # Servicer velocity

# Target parameters (free joint with 7 parameters: 3 pos, 4 quat)
target_offset = 7
data.qpos[target_offset:target_offset+3] = np.array([3, 0, 0])
data.qvel[target_offset:target_offset+3] = np.array([0, 0, 0])

# Simulation parameters
duration = 5.0    # seconds
dt = model.opt.timestep
steps = int(duration / dt)

# Arrays to store positions for each satellite
servicer_positions = np.zeros((steps, 3))
target_positions = np.zeros((steps, 3))

# Run simulation and record positions
for i in range(steps):
    servicer_positions[i] = data.qpos[0:3]
    target_positions[i] = data.qpos[target_offset:target_offset+3]
    mujoco.mj_step(model, data)

# ----- Animation Setup -----
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Create plot objects for satellites (as markers) and trajectories (as dashed lines)
servicer_point, = ax.plot([], [], [], 'o', color='red', markersize=15, label='Servicer')
target_point, = ax.plot([], [], [], 'o', color='blue', markersize=15, label='Target')
servicer_traj, = ax.plot([], [], [], '--', color='red', linewidth=1)
target_traj, = ax.plot([], [], [], '--', color='blue', linewidth=1)

# Set axis limits based on simulation data (adding a margin)
margin = 1.0
ax.set_xlim(servicer_positions[:,0].min()-margin, servicer_positions[:,0].max()+margin)
ax.set_ylim(servicer_positions[:,1].min()-margin, servicer_positions[:,1].max()+margin)
ax.set_zlim(servicer_positions[:,2].min()-margin, servicer_positions[:,2].max()+margin)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Satellite Shapes Animation')
ax.legend()

# ----- Animation Functions -----
def init():
    # Initialize plot objects to empty data
    servicer_point.set_data([], [])
    servicer_point.set_3d_properties([])
    target_point.set_data([], [])
    target_point.set_3d_properties([])
    servicer_traj.set_data([], [])
    servicer_traj.set_3d_properties([])
    target_traj.set_data([], [])
    target_traj.set_3d_properties([])
    return servicer_point, target_point, servicer_traj, target_traj

def update(frame):
    # Update marker positions for servicer and target
    sx, sy, sz = servicer_positions[frame]
    tx, ty, tz = target_positions[frame]
    servicer_point.set_data([sx], [sy])
    servicer_point.set_3d_properties([sz])
    target_point.set_data([tx], [ty])
    target_point.set_3d_properties([tz])
    
    # Update trajectory lines for both satellites up to current frame
    servicer_traj.set_data(servicer_positions[:frame+1, 0], servicer_positions[:frame+1, 1])
    servicer_traj.set_3d_properties(servicer_positions[:frame+1, 2])
    target_traj.set_data(target_positions[:frame+1, 0], target_positions[:frame+1, 1])
    target_traj.set_3d_properties(target_positions[:frame+1, 2])
    
    return servicer_point, target_point, servicer_traj, target_traj

# Create animation using FuncAnimation
anim = animation.FuncAnimation(fig, update, frames=steps, init_func=init, interval=20, blit=True)

# Save the animation as an MP4 video file (requires ffmpeg)
anim.save('output/satellite_animation.mp4', writer='ffmpeg', fps=1/dt)

plt.show()
