# sanity_check_fixed.py

import numpy as np
import mujoco
from .satellite_marl_env import raw_env

def main():
    # Create the env
    env = raw_env()

    # Zero out damping/friction for clarity
    try:
        env.model.dof_damping[:]    = 0.0
        env.model.geom_friction[:]  = 0.0
        print("[DEBUG] Damping & friction zeroed.")
    except:
        print("[DEBUG] Could not zero damping/friction.")

    # Fetch IDs
    env._get_mujoco_ids()
    print(f"[DEBUG] Body IDs: {env.body_ids}")
    print(f"[DEBUG] xfrc_applied shape: {env.data.xfrc_applied.shape}")

    # Reset and measure initial distance
    obs, info = env.reset()
    dist0, _, _ = env._get_current_state_metrics()
    print(f"[DEBUG] Initial distance: {dist0:.4f} m")

    # Loop applying *inverted* chase thrust
    for step in range(200):
        # 1) compute world-frame vector from servicer → target
        p_s = env.data.site_xpos[env.site_ids["servicer_dock"]]
        p_t = env.data.site_xpos[env.site_ids["target_dock"]]
        vec = p_t - p_s
        norm = np.linalg.norm(vec) or 1.0

        # 2) INVERTED unit thrust (–veĉ) so that positive force moves toward the port
        thrust_dir = -vec / norm
        force = thrust_dir * 5.0     # 5 N for a stronger effect

        # 3) build action array: [Fx,Fy,Fz, 0,0,0]
        action = np.concatenate([force, np.zeros(3)])
        actions = {"servicer": action, "target": np.zeros(6)}

        # 4) apply & log
        env._apply_actions(actions)
        idx = env.body_ids["servicer"] - 1
        print(f"Step {step:3d} | xfrc_applied = {env.data.xfrc_applied[idx]}")

        # 5) forward & log vel
        mujoco.mj_forward(env.model, env.data)
        vel = env.data.qvel[
            env.joint_qvel_adr["servicer"]:env.joint_qvel_adr["servicer"]+3
        ]
        print(f"        post-forward vel = {vel}")

        # 6) mj_step + distance
        _, _, terms, _, _ = env.step(actions)
        dist, _, _ = env._get_current_state_metrics()
        print(f"        distance = {dist:.4f} m   terminated={terms['__all__']}")
        if terms["__all__"]:
            print(f"[DEBUG] Terminated at step {step}.")
            break

    env.close()

if __name__ == "__main__":
    main()
