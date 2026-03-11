import sys
import os
import argparse
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments for Isaac Lab AppLauncher
parser = argparse.ArgumentParser(description="Yoink RL Evaluation")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments for evaluation")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file to load")
parser.add_argument("--tracking_weight", type=float, default=3.0, help="Weight for distance tracking reward")
parser.add_argument("--alignment_weight", type=float, default=0.0, help="Weight for facing the target reward")
parser.add_argument("--success_weight", type=float, default=5.0, help="Weight for reaching the target")
parser.add_argument("--collision_weight", type=float, default=-1.0, help="Weight for hitting obstacles")
parser.add_argument("--video", action="store_true", help="Record mp4 video of the evaluation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from src.model.isaac_env import EnvCfg
from src.model.yoink import PPO

def main():
    # 1. Setup Env
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Apply Curriculum Weights
    env_cfg.rewards["tracking"].weight = args_cli.tracking_weight
    env_cfg.rewards["alignment"].weight = args_cli.alignment_weight
    env_cfg.rewards["success"].weight = args_cli.success_weight
    env_cfg.rewards["collision"].weight = args_cli.collision_weight
    
    # Force enable cameras if video recording is requested
    if args_cli.video:
        args_cli.enable_cameras = True
        
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap environment for video recording if requested
    if args_cli.video:
        video_dir = os.path.join(os.path.dirname(__file__), "videos")
        print(f"Recording evaluation video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, video_dir, step_trigger=lambda step: step == 0, video_length=500)
    
    # 2. Setup Model
    state_dim = 28
    action_dim = 2
    
    ppo_agent = PPO(state_dim, action_dim, lr=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2)
    
    # Load Checkpoint
    checkpoint_path = args_cli.checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        ppo_agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded checkpoint from: {checkpoint_path} (Step: {checkpoint.get('step', 'unknown')})")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        simulation_app.close()
        return

    # Set policy to evaluation mode to disable exploratory dropout/noise
    ppo_agent.policy.eval()

    # 3. Evaluation Loop
    obs, _ = env.reset()
    current_obs = obs["policy"] # Shape: [num_envs, 28]
    current_obs = torch.nan_to_num(current_obs, nan=0.0, posinf=1.0, neginf=-1.0)

    print("Starting evaluation loop...")
    
    # 3.5 Setup Debug UI Overlay Native to Isaac Sim
    import omni.ui as ui
    window = ui.Window("NN Input Debugger", width=400, height=500, visible=True)
    with window.frame:
        with ui.VStack():
            ui.Label("NEURAL NETWORK INPUT TENSOR", style={"font_size": 20, "color": 0xFF00FF00})
            ui.Spacer(height=10)
            lbl_imu = ui.Label("IMU Input [X-Vel, Yaw-Vel]:", style={"font_size": 16})
            lbl_target = ui.Label("Target Input [Local-X, Local-Y]:", style={"font_size": 16})
            lbl_lidar_hdr = ui.Label("Lidar Inputs (24 Rays Normalize [0-1]):", style={"font_size": 16})
            lbl_lidar = ui.Label("", style={"font_size": 12})
    
    while simulation_app.is_running():
        # --- Update HUD ---
        # Get the first environment's observation vector
        obs_vec = current_obs[0].cpu().numpy()
        lbl_imu.text = f"IMU Input [X-Vel, Yaw-Vel]: [{obs_vec[0]:.3f}, {obs_vec[1]:.3f}]"
        lbl_target.text = f"Target Input [Local-X, Local-Y]: [{obs_vec[2]:.3f}, {obs_vec[3]:.3f}]"
        # Format the 24 lidar rays into a grid string
        lidar_str = ""
        for i in range(4, 28):
            lidar_str += f"{obs_vec[i]:.2f}  "
            if (i - 4 + 1) % 6 == 0:
                lidar_str += "\n"
        lbl_lidar.text = lidar_str
        
        with torch.no_grad():
            # For evaluation, we directly output the mean action (deterministic) 
            # rather than using sample_action to prevent exploratory noise.
            action_mean, _ = ppo_agent.policy(current_obs)

        # Enforce action limits
        nn_actions = torch.clamp(action_mean, -1.0, 1.0)
        
        # --- Map 2 Outputs to 4 Wheels ---
        left_cmd = nn_actions[:, 0].unsqueeze(1)  # Shape [N, 1]
        right_cmd = nn_actions[:, 1].unsqueeze(1) # Shape [N, 1]
        env_actions = torch.cat([left_cmd, right_cmd, left_cmd, right_cmd], dim=1)
        
        # --- Step Environment ---
        obs, rewards, terms, truncs, infos = env.step(env_actions)
        
        # If environments terminate, they automatically reset in IsaacLab ManagerBasedRLEnv
        # We just need to grab the new observations
        current_obs = obs["policy"]
        current_obs = torch.nan_to_num(current_obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
    env.close()

if __name__ == "__main__":
    main()
