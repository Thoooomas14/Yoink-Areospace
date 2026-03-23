import sys
import os
import argparse
import math
import numpy as np
import torch
import traceback

from isaaclab.app import AppLauncher

# --- 1. CLI Setup ---
parser = argparse.ArgumentParser(description="Lynxmotion A4WD1 SB3 Evaluation with Live Debugging")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--checkpoint", type=str, help="Path to the SB3 checkpoint .zip file")
parser.add_argument("--tracking_weight",  type=float, default=0.2)
parser.add_argument("--motion_weight",  type=float, default=0.4)
parser.add_argument("--collision_weight", type=float, default=0.4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Imports (Post-App Launch) ---
import omni.ui as ui
import omni.timeline
from stable_baselines3 import PPO
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from src.model.isaac_env import EnvCfg
from src.model.yoink import get_ppo_agent


# Global state for interactive target control
TARGET_CONTROL_STATE = {
    "pos": [0.0, 0.0, 0.2],  # default [x, y, z]
    "needs_update": False
}


def build_ui():
    """Build the debug UI window.
    Returns (window, labels) — caller MUST hold 'window' reference or it gets GC'd.
    """
    window = ui.Window("Lynxmotion SB3 Debugger", width=440, height=500)
    labels = {}
    with window.frame:
        with ui.VStack(spacing=8, style={"padding": 14}):
            ui.Label("LIVE SB3 NEURAL NETWORK I/O",
                     style={"font_size": 20, "color": 0xFF00FF00})
            ui.Line(style={"color": 0xFF555555})

            ui.Label("── OBSERVATIONS ──",
                     style={"font_size": 13, "color": 0xFF00AAFF})
            labels["imu"]    = ui.Label("IMU  [LinVel-X, YawRate]: ---",
                                        style={"font_size": 13, "color": 0xFFFFFFFF})
            labels["target"] = ui.Label("Goal [Dist-norm, Angle]:  ---",
                                        style={"font_size": 13, "color": 0xFFFFFFFF})

            ui.Spacer(height=4)
            ui.Label("── LIDAR (0.0 = close, 1.0 = 10 m+) ──",
                     style={"font_size": 12, "color": 0xFFAAAAAA})
            labels["lidar"]  = ui.Label("---",
                                        style={"font_size": 11, "color": 0xFFCCCCCC,
                                               "word_wrap": True})

            ui.Label("── LIDAR  (real metres) ──",
                     style={"font_size": 12, "color": 0xFFAAAAAA})
            labels["lidar_raw"] = ui.Label("---",
                                            style={"font_size": 11, "color": 0xFFFFAA55,
                                                   "word_wrap": True})

            ui.Line(style={"color": 0xFF555555})
            ui.Label("── ACTIONS / POLICY OUTPUT ──",
                     style={"font_size": 13, "color": 0xFFFFCC00})
            labels["action"] = ui.Label("Twist [v, omega]: ---",
                                        style={"font_size": 16, "color": 0xFFFFFFFF})
            labels["action_raw"] = ui.Label("Policy Raw Output: [---, ---]",
                                         style={"font_size": 11, "color": 0xFFAAAA00})

            ui.Line(style={"color": 0xFF555555})
            ui.Label("── REWARDS ──",
                     style={"font_size": 13, "color": 0xFFFF4444})
            labels["rew_step"]    = ui.Label("Step  reward: ---",
                                             style={"font_size": 13, "color": 0xFFFF8888})
            labels["rew_episode"] = ui.Label("Episode sum: ---",
                                             style={"font_size": 13, "color": 0xFFFF5555})

            ui.Line(style={"color": 0xFF555555})
            ui.Label("── MANUAL TARGET CONTROL ──",
                     style={"font_size": 13, "color": 0xFFFF00FF})
            
            with ui.HStack(height=20):
                ui.Label("Target X:", width=70)
                x_slider = ui.FloatSlider(min=-7.0, max=7.0)
                x_slider.model.set_value(TARGET_CONTROL_STATE["pos"][0])
                def on_x_change(m):
                    TARGET_CONTROL_STATE["pos"][0] = m.as_float
                    TARGET_CONTROL_STATE["needs_update"] = True
                x_slider.model.add_value_changed_fn(on_x_change)

            with ui.HStack(height=20):
                ui.Label("Target Y:", width=70)
                y_slider = ui.FloatSlider(min=-7.0, max=7.0)
                y_slider.model.set_value(TARGET_CONTROL_STATE["pos"][1])
                def on_y_change(m):
                    TARGET_CONTROL_STATE["pos"][1] = m.as_float
                    TARGET_CONTROL_STATE["needs_update"] = True
                y_slider.model.add_value_changed_fn(on_y_change)

            def on_reset_btn():
                TARGET_CONTROL_STATE["needs_update"] = "RANDOM"
                
            ui.Button("Force RANDOM Target Reset", clicked_fn=on_reset_btn, height=30)

    return window, labels


def main():
    # --- 3. Environment Setup ---
    env_cfg = EnvCfg()
    # Apply weights to config
    env_cfg.rewards["tracking"].weight  = args_cli.tracking_weight
    env_cfg.rewards["motion"].weight  = args_cli.motion_weight
    env_cfg.rewards["collision"].weight = args_cli.collision_weight

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.lidar.debug_vis = True

    # --- Terrain Diversity Calculation ---
    # Aim for 16 robots per terrain patch
    num_terrains = max(1, args_cli.num_envs // 16)
    n_rows = int(math.sqrt(num_terrains))
    n_cols = num_terrains // n_rows
    env_cfg.scene.terrain.terrain_generator.num_rows = n_rows
    env_cfg.scene.terrain.terrain_generator.num_cols = n_cols
    print(f"[eval] Configuring terrain grid: {n_rows}x{n_cols} ({num_terrains} unique patches)")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    if args_cli.checkpoint is None:
        print("WARNING: No checkpoint provided. Running with RANDOM WEIGHTS.")
        model = get_ppo_agent(env)
    elif args_cli.checkpoint.endswith(".pt"):
        print(f"Loading raw PyTorch weights from: {args_cli.checkpoint}")
        model = get_ppo_agent(env)
        checkpoint = torch.load(args_cli.checkpoint, map_location="cuda")
        state_dict = checkpoint["policy_state_dict"] if "policy_state_dict" in checkpoint else checkpoint
        model.policy.load_state_dict(state_dict)
    else:
        print(f"Loading SB3 checkpoint: {args_cli.checkpoint}")
        model = PPO.load(args_cli.checkpoint, env=env, device="cuda")

    # --- 5. UI Setup ---
    # IMPORTANT: hold 'debug_window' here so it is NOT garbage collected.
    debug_window, lbl = build_ui()

    # --- 6. Simulation Start ---
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()  # Prime the app so is_running() returns True

    obs = env.reset()        # Sb3VecEnvWrapper already returns plain numpy (num_envs, obs_dim)
    print(f"[eval] obs shape: {obs.shape}  |  dtype: {obs.dtype}")
    
    print("[eval] Entering loop. Press Ctrl+C to stop.")

    count = 0
    ep_return = 0.0  # running sum of rewards for the current episode
    while simulation_app.is_running():
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        v = obs[0]  # (obs_dim,) for env 0

        # --- Update UI labels via .text property ---
        lbl["imu"].text    = f"IMU  [LinVel-X, YawRate]:  {v[0]:+.3f}  {v[1]:+.3f}"
        lbl["target"].text = f"Goal [Dist-norm, Angle]:   {v[2]:+.3f}  {v[3]:+.3f}"

        rays = v[4:]   # 24 lidar rays (360° / 15° res)
        cols = 6
        rows = [rays[i : i + cols] for i in range(0, len(rays), cols)]
        lbl["lidar"].text     = "\n".join("  ".join(f"{x:.2f}"       for x in row) for row in rows)
        rows_m = [rays[i : i + cols] * 10.0 for i in range(0, len(rays), cols)]
        lbl["lidar_raw"].text = "\n".join("  ".join(f"{x:.2f} m"    for x in row) for row in rows_m)

        # Policy prediction
        actions, _ = model.predict(obs, deterministic=True)
        lbl["action"].text = (
            f"Twist  [v={actions[0,0]:+.2f}   omega={actions[0,1]:+.2f}]"
        )
        # Show raw output from model before lin_scale/ang_scale
        lbl["action_raw"].text = f"Policy [v_raw={actions[0,0]:+.3f}   w_raw={actions[0,1]:+.3f}]"

        # --- Step ---
        obs, rewards, dones, infos = env.step(actions)

        # --- Reward display ---
        step_rew = float(rewards[0])
        ep_return += step_rew
        lbl["rew_step"].text    = f"Step  reward:  {step_rew:+.4f}"
        lbl["rew_episode"].text = f"Episode sum:   {ep_return:+.4f}"
        if dones[0]:
            ep_return = 0.0  # reset accumulator for next episode

        # --- Interactive Target Control Logic ---
        # We access the underlying Isaac Lab environment via .unwrapped
        base_env = env.unwrapped
        target_asset = base_env.scene["target"]
        
        if TARGET_CONTROL_STATE["needs_update"] == "RANDOM":
            # Trigger the standard random reset event for all envs
            from src.model.isaac_env import reset_target_state_from_terrain
            from isaaclab.managers import SceneEntityCfg
            reset_target_state_from_terrain(base_env, torch.arange(base_env.num_envs, device=base_env.device),
                                           pose_range={}, velocity_range={})
            TARGET_CONTROL_STATE["needs_update"] = False
            # Update the sliders state to match current pos (optional, but good for UI)
            curr_pos = target_asset.data.root_pos_w[0]
            TARGET_CONTROL_STATE["pos"][0] = curr_pos[0].item()
            TARGET_CONTROL_STATE["pos"][1] = curr_pos[1].item()
            # Note: We can't easily push these back to UI models without storing references to x_slider/y_slider.
            # For now, it's enough to reset the internal state.

        elif TARGET_CONTROL_STATE["needs_update"]:
            target_pos = torch.zeros((base_env.num_envs, 7), device=base_env.device)
            # Default root state + new XY
            target_pos[:, :7] = target_asset.data.default_root_state[:, :7]
            target_pos[:, 0] = TARGET_CONTROL_STATE["pos"][0]
            target_pos[:, 1] = TARGET_CONTROL_STATE["pos"][1]
            target_pos[:, 2] = TARGET_CONTROL_STATE["pos"][2]
            
            target_asset.write_root_pose_to_sim(target_pos, env_ids=torch.arange(base_env.num_envs, device=base_env.device))
            target_asset.write_root_velocity_to_sim(torch.zeros((base_env.num_envs, 6), device=base_env.device), 
                                                 env_ids=torch.arange(base_env.num_envs, device=base_env.device))
            TARGET_CONTROL_STATE["needs_update"] = False

        # --- Flush UI + viewport (must be last) ---
        simulation_app.update()

        count += 1
        if count % 100 == 0:
            print(f"[eval] step={count:5d}  reward={rewards[0]:+.4f}  done={dones[0]}"
                  f"  actions={np.round(actions[0], 3)}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n====== EVAL CRASHED — FULL TRACEBACK ======")
        traceback.print_exc()
        print("===========================================\n")
        sys.exit(1)