import sys
import os
import argparse
import numpy as np
import torch
import traceback

from isaaclab.app import AppLauncher

# --- 1. CLI Setup ---
parser = argparse.ArgumentParser(description="Lynxmotion A4WD1 SB3 Evaluation with Live Debugging")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--checkpoint", type=str, help="Path to the SB3 checkpoint .zip file")
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

            ui.Line(style={"color": 0xFF555555})
            ui.Label("── ACTIONS ──",
                     style={"font_size": 13, "color": 0xFFFFCC00})
            labels["action"] = ui.Label("Wheels [FL, RL, FR, RR]: ---",
                                        style={"font_size": 16, "color": 0xFFFFFFFF})

    return window, labels


def main():
    # --- 3. Environment Setup ---
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.lidar.debug_vis = True

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    # --- 4. Model Setup ---
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[128, 32], vf=[128, 32]))

    if args_cli.checkpoint is None:
        print("WARNING: No checkpoint provided. Running with RANDOM WEIGHTS.")
        model = PPO("MlpPolicy", env, device="cuda", policy_kwargs=policy_kwargs)
    elif args_cli.checkpoint.endswith(".pt"):
        print(f"Loading raw PyTorch weights from: {args_cli.checkpoint}")
        model = PPO("MlpPolicy", env, device="cuda", policy_kwargs=policy_kwargs)
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
    while simulation_app.is_running():
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        v = obs[0]  # (obs_dim,) for env 0

        # --- Update UI labels via .text property ---
        lbl["imu"].text    = f"IMU  [LinVel-X, YawRate]:  {v[0]:+.3f}  {v[1]:+.3f}"
        lbl["target"].text = f"Goal [Dist-norm, Angle]:   {v[2]:+.3f}  {v[3]:+.3f}"

        rays = v[4:]   # 23 lidar rays
        cols = 6
        rows = [rays[i : i + cols] for i in range(0, len(rays), cols)]
        lbl["lidar"].text = "\n".join("  ".join(f"{x:.2f}" for x in row) for row in rows)

        actions, _ = model.predict(obs, deterministic=True)
        lbl["action"].text = (
            f"Skid-steer  [Left={actions[0,0]:+.2f}   Right={actions[0,1]:+.2f}]"
        )

        # --- Step ---
        obs, rewards, dones, infos = env.step(actions)

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