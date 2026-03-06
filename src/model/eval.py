import sys
import os
import argparse
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments for Isaac Lab AppLauncher
parser = argparse.ArgumentParser(description="Yoink RL Evaluation")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments for evaluation")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file to load")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from src.model.isaac_env import EnvCfg
from src.model.yoink import PPO

def main():
    # 1. Setup Env
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
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
    current_obs = torch.nan_to_num(current_obs, nan=0.0, posinf=0.0, neginf=0.0)

    print("Starting evaluation loop...")
    while simulation_app.is_running():
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
        current_obs = torch.nan_to_num(current_obs, nan=0.0, posinf=0.0, neginf=0.0)
            
    env.close()

if __name__ == "__main__":
    main()
