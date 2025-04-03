import os
import platform
# Set appropriate backend for macOS
if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "glfw"  # macOS native OpenGL context
else:
    os.environ["MUJOCO_GL"] = "osmesa"  # For Linux systems

import mujoco
import numpy as np
import time
import imageio.v2 as imageio
import matplotlib.pyplot as plt  # For debugging
import mediapy as media

# Create output directory
os.makedirs("output", exist_ok=True)

# XML model definition
sat_xml = """<mujoco model="satellites">
  <compiler angle="degree" />
  <option timestep="0.01" gravity="0 0 0" />
  <default>
    <!-- Default geom properties -->
    <geom contype="1" conaffinity="1" friction="0.1" density="1000"/>
  </default>

  <worldbody>
    <!-- Create a camera that follows the servicer -->
    <camera name="trackercam" mode="trackcom" target="servicer" pos="0 -3 0"/>
    <!-- Fixed camera -->
    <camera name="fixed" pos="5 0 0" xyaxes="-1 0 0 0 1 0"/>
    
    <!-- Lighting -->
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
    
    <!-- Servicer Satellite -->
    <body name="servicer" pos="0 0 0">
      <freejoint/>
      <!-- Main body (e.g., sphere) -->
      <geom type="sphere" size="0.5" rgba="0.5 0.5 0.5 1"/>
      <!-- Docking port protrusion -->
      <geom name="servicer_dock_port" type="cylinder" pos="0 0 0.5" size="0.1 0.2" euler="0 0 0" rgba="1 0 0 1"/>
      <!-- Docking detection site -->
      <site name="servicer_dock_site" pos="0 0 0.5" size="0.15" rgba="0 1 0 0.5" />
    </body>

    <!-- Target Satellite -->
    <body name="target" pos="2 0 0">
      <freejoint/>
      <!-- Main body (e.g., sphere) -->
      <geom type="sphere" size="0.5" rgba="0.5 0.5 0.5 1"/>
      <!-- Docking cavity (for receiving the docking port) -->
      <geom name="target_dock_cavity" type="cylinder" pos="0 0 -0.5" size="0.15 0.25" euler="0 0 0" rgba="0 0 1 1"/>
      <!-- Docking detection site -->
      <site name="target_dock_site" pos="0 0 -0.5" size="0.15" rgba="0 1 0 0.5" />
    </body>
  </worldbody>

  <tendon>
    <!-- Optional: Tendon to visualize distance between docking sites -->
    <spatial limited="false" width="0.01" rgba="1 1 0 0.5">
      <site site="servicer_dock_site"/>
      <site site="target_dock_site"/>
    </spatial>
  </tendon>

  <visual>
    <global offwidth="1920" offheight="1080" />
  </visual>
</mujoco>
"""

# Load the model
model = mujoco.MjModel.from_xml_string(sat_xml)

# Create simulation data
data = mujoco.MjData(model)

# Set initial conditions
# Servicer position and velocity
data.qpos[0:3] = np.array([0, 0, 0])      # x, y, z position
data.qpos[3:7] = np.array([1, 0, 0, 0])   # quaternion orientation
data.qvel[0:3] = np.array([1, 0, 0])      # velocity

# Target position and velocity (7 DOF offset for the second free joint)
target_offset = 7
data.qpos[target_offset:target_offset+3] = np.array([3, 0, 0])  # position
data.qpos[target_offset+3:target_offset+7] = np.array([1, 0, 0, 0])  # orientation
data.qvel[target_offset:target_offset+3] = np.array([0, 0, 0])  # velocity

# Simulation parameters
duration = 15.0  # seconds
dt = model.opt.timestep
fps = 30  # Frames per second for rendering
render_every = int(1.0 / (dt * fps))  # How many simulation steps between renders

# Setup for recording
record = True
video_path = os.path.join("output", "satellite_simulation.mp4")
width, height = 640, 480

# Option 1: Interactive viewer without recording
def run_interactive():
    # Initialize the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial sync
        viewer.sync()
        
        # Run simulation with rendering
        step_count = 0
        start_time = time.time()
        sim_time = 0
        
        while sim_time < duration:
            # Step the simulation
            mujoco.mj_step(model, data)
            sim_time += dt
            step_count += 1
            
            # Only render at the specified FPS
            if step_count % render_every == 0:
                # Update visualization
                viewer.sync()
                
                # Optional: add time delay to watch in real-time
                elapsed = time.time() - start_time
                if elapsed < sim_time:
                    time.sleep(sim_time - elapsed)
        
        # Keep the viewer open for a moment to see the final state
        time.sleep(1.0)

# Option 2: Render to offscreen buffer and save video
duration = 3.8  # (seconds)
framerate = 60  # (Hz)
frames = []
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
mujoco.mj_resetData(model, data)
with mujoco.Renderer(model) as renderer:
  while data.time < duration:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data, scene_option=scene_option)
      pixels = renderer.render()
      frames.append(pixels)

media.show_video(frames, fps=framerate)

# Choose which method to use


print("Simulation complete!")
'''
with mujoco.Renderer(model) as renderer:
  mujoco.mj_forward(model, data)
  renderer.update_scene(data)

  media.show_image(renderer.render())
  '''