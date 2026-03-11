import sys
import os
import argparse
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments for Isaac Lab AppLauncher
parser = argparse.ArgumentParser(description="Yoink RL Training")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments (use 4096 for headless training)")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
parser.add_argument("--tracking_weight", type=float, default=3.0, help="Weight for distance tracking reward")
parser.add_argument("--alignment_weight", type=float, default=0.0, help="Weight for facing the target reward")
parser.add_argument("--success_weight", type=float, default=5.0, help="Weight for reaching the target")
parser.add_argument("--collision_weight", type=float, default=-2.0, help="Weight for hitting obstacles")
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
    
    # Apply Curriculum Weights
    env_cfg.rewards["tracking"].weight = args_cli.tracking_weight
    env_cfg.rewards["alignment"].weight = args_cli.alignment_weight
    env_cfg.rewards["success"].weight = args_cli.success_weight
    env_cfg.rewards["collision"].weight = args_cli.collision_weight
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 2. Setup Model
    state_dim = 28
    action_dim = 2
    
    ppo_agent = PPO(state_dim, action_dim, lr=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2)
    
    update_timestep = 200 # update policy every 200 timesteps
    max_training_timesteps = 500000

    time_step = 0
    i_episode = 0
    
    # 2.5 Resume from checkpoint if provided
    if args_cli.resume:
        if os.path.exists(args_cli.resume):
            print(f"Loading checkpoint from: {args_cli.resume}")
            checkpoint = torch.load(args_cli.resume)
            ppo_agent.policy.load_state_dict(checkpoint["policy_state_dict"])
            ppo_agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])
            ppo_agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            time_step = checkpoint["step"]
            print(f"Resumed training at step {time_step}")
        else:
            print(f"Error: Checkpoint file {args_cli.resume} does not exist. Starting from scratch.")

    # 4. Checkpoint setup
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_every = 2000  # save every 2k steps

    # 3. Training Loop
    obs, _ = env.reset()
    current_obs = obs["policy"] # Shape: [num_envs, 28]
    current_obs = torch.nan_to_num(current_obs, nan=0.0, posinf=1.0, neginf=-1.0)

    while simulation_app.is_running() and time_step <= max_training_timesteps:
        # --- A. Get NN Actions (Shape: [N, 2]) ---
        # The output is mapped between -1 and 1 via Tanh.
        nn_actions = ppo_agent.select_action(current_obs) 

        # --- B. Map 2 Outputs to 4 Wheels ---
        left_cmd = nn_actions[:, 0].unsqueeze(1)  # Shape [N, 1]
        right_cmd = nn_actions[:, 1].unsqueeze(1) # Shape [N, 1]
        env_actions = torch.cat([left_cmd, right_cmd, left_cmd, right_cmd], dim=1)
        
        # --- C. Step Environment ---
        obs, rewards, terms, truncs, infos = env.step(env_actions)
        
        # --- D. Training Step ---
        ppo_agent.buffer.rewards.append(rewards)
        is_terminals = terms | truncs
        ppo_agent.buffer.dones.append(is_terminals)
        
        time_step += 1
        current_obs = obs["policy"]
        current_obs = torch.nan_to_num(current_obs, nan=0.0, posinf=1.0, neginf=-1.0)

        if time_step % update_timestep == 0:
            ppo_agent.update()

        if time_step % 100 == 0:
            print(f"Step {time_step} | Avg Reward: {rewards.mean().item():.4f}")

        # --- E. Save Checkpoint ---
        if time_step % save_every == 0:
            path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
            torch.save({
                "step": time_step,
                "policy_state_dict": ppo_agent.policy.state_dict(),
                "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
            }, path)
            print(f"Checkpoint saved: {path}")

    # Save final checkpoint
    path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    torch.save({
        "step": time_step,
        "policy_state_dict": ppo_agent.policy.state_dict(),
        "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
    }, path)
    print(f"Final checkpoint saved: {path}")
            
    env.close()

if __name__ == "__main__":
    main()
