import sys
import os
import argparse
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments for Isaac Lab AppLauncher
parser = argparse.ArgumentParser(description="Yoink RL Training")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments (use 4096 for headless training)")
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
    
    update_timestep = 2000 # update policy every 2000 timesteps
    max_training_timesteps = 500000

    time_step = 0
    i_episode = 0

    # 4. Checkpoint setup
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_every = 10000  # save every 10k steps

    # 3. Training Loop
    obs, _ = env.reset()
    current_obs = obs["policy"] # Shape: [num_envs, 28]
    current_obs = torch.nan_to_num(current_obs, nan=0.0, posinf=0.0, neginf=0.0)

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
        current_obs = torch.nan_to_num(current_obs, nan=0.0, posinf=0.0, neginf=0.0)

        if time_step % update_timestep == 0:
            ppo_agent.update()

        if time_step % 100 == 0:
            print(f"Step {time_step} | Avg Reward: {rewards.mean().item():.4f}")

        # --- E. Save Checkpoint ---
        if time_step % save_every == 0:
            path = os.path.join(checkpoint_dir, f"checkpoint_{time_step}.pt")
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
