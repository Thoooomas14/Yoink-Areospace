import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Visualize Terrain")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless rendering to circumvent the UI crash
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from src.model.isaac_env import EnvCfg

def main():
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = 2 # 2 robots is enough to see the terrain
    
    # render_mode must be rgb_array for gymnasium's RecordVideo wrapper
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    
    videos_dir = os.path.join(os.path.dirname(__file__), "videos")
    env = gym.wrappers.RecordVideo(
        env,
        videos_dir,
        step_trigger=lambda step: step == 0,
        video_length=150,
        disable_logger=True,
    )

    env.reset()
    
    print(f"[INFO] Recording a 150-frame video to {videos_dir}...")
    
    with torch.inference_mode():
        for _ in range(150):
            # Command random velocities to wheel joints so the rovers bounce off rocks
            actions = torch.randn((2, 4), device=env.unwrapped.device) * 2.0
            env.step(actions)
            
    print("[INFO] Video Recording complete!")
    env.close()

if __name__ == "__main__":
    main()
