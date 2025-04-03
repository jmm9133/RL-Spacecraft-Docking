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
def render_offscreen():
    # Initialize renderer with proper settings
    # Make sure camera exists in the model
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "trackercam")
    if camera_id == -1:
        print("Warning: 'trackercam' not found, falling back to first camera")
        camera_name = None  # Will use the first camera if trackercam not found
    else:
        camera_name = "trackercam"
    
    # Initialize with more explicit settings
    renderer = mujoco.Renderer(model, height=height, width=width)
    # Additional settings to ensure proper rendering
    renderer.enable_skybox(True)
    renderer.set_flags(
        scene_option=mujoco.mjtRndFlag.mjRND_SHADOW,
        flags={"wireframe": False}
    )
    
    # Prepare for recording frames
    frames = []
    step_count = 0
    
    # Run simulation with rendering
    while step_count * dt < duration:
        # Step the simulation
        mujoco.mj_step(model, data)
        step_count += 1
        
        # Only render at the specified FPS
        if step_count % render_every == 0:
            # Make sure the camera is properly attached and scene is updated
            mujoco.mj_forward(model, data)  # This ensures all MuJoCo computations are done
            
            # Update scene and render
            renderer.update_scene(data, camera=camera_name)
            
            # Render with more explicit options
            pixels = renderer.render()
            
            # Debug: Save the first frame to inspect it
            if step_count == render_every:
                debug_img_path = os.path.join("output", "debug_first_frame.png")
                plt.imsave(debug_img_path, pixels)
                print(f"Saved debug frame to {debug_img_path}")
                print(f"Frame shape: {pixels.shape}, min: {pixels.min()}, max: {pixels.max()}, dtype: {pixels.dtype}")
            
            # Convert to uint8 if needed
            if pixels.dtype != np.uint8:
                pixels = (pixels * 255).astype(np.uint8)
            
            # Add to frames
            frames.append(pixels)
    
    # Save the video if frames were captured
    if frames:
        print(f"Saving {len(frames)} frames to {video_path}")
        imageio.mimsave(video_path, frames, fps=fps)

# Choose which method to use
if record:
    try:
        # Add additional debugging information
        print("Starting offscreen rendering...")
        render_offscreen()
        print(f"Video saved to {video_path}")
        
        # Verify the video file was created with some content
        if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
            print(f"Video file created successfully with size: {os.path.getsize(video_path)} bytes")
        else:
            print("Warning: Video file may be empty or corrupt")
    except Exception as e:
        print(f"Error during offscreen rendering: {e}")
        print("Falling back to interactive mode without recording")
        run_interactive()
else:
    run_interactive()

print("Simulation complete!")