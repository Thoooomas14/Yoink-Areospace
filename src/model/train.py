import sys
import os
import argparse
import traceback
import torch

from isaaclab.app import AppLauncher

# --- 1. CLI Setup ---
parser = argparse.ArgumentParser(description="Yoink RL Training with Stable Baselines3")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
parser.add_argument("--max_steps", type=int, default=1000000, help="Total timesteps to train")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to load a model to resume training")
# Reward weights — must be POSITIVE. The reward functions themselves return
# negative values for bad behaviour; the weight just scales the magnitude.
parser.add_argument("--tracking_weight",  type=float, default=0.8)
parser.add_argument("--collision_weight", type=float, default=0.2)
parser.add_argument("--save_interval", type=int, default=25_000_000,
                    help="Save checkpoint every N timesteps (default 25M ≈ 25 min at 17K fps)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Post-App Launch Imports ---
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from src.model.isaac_env import EnvCfg


class OverwriteCheckpointCallback(BaseCallback):
    """Saves the model to a single fixed file, overwriting it each time.
    save_interval is in TIMESTEPS (not rollout steps), so it behaves
    consistently regardless of num_envs.
    """

    def __init__(self, save_interval: int, save_path: str, filename: str = "checkpoint_progress", verbose: int = 1):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path
        self.filename = filename
        self._last_save_timestep = 0

    def _on_training_start(self) -> None:
        self._last_save_timestep = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_timestep >= self.save_interval:
            self._last_save_timestep = self.num_timesteps
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(self.save_path, self.filename)
            self.model.save(path)
            if self.verbose:
                print(f"[train] Saved checkpoint → {path}.zip  (step {self.num_timesteps:,})")
        return True


def main():
    # --- 3. Setup Environment ---
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Apply weights to config
    env_cfg.rewards["tracking"].weight = args_cli.tracking_weight
    env_cfg.rewards["collision"].weight = args_cli.collision_weight

    # Create the environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # Wrap for SB3 compatibility (handles tensor conversion and normalization)
    env = Sb3VecEnvWrapper(env)

    # --- 4. Setup PPO Model ---
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[128, 32], vf=[128, 32])  # pi=actor, vf=critic
    )

    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        print(f"Resuming training from: {args_cli.checkpoint}")
        model = PPO.load(args_cli.checkpoint, env=env, device="cuda")
    else:
        print("Starting training from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,       # Buffer size per env per update
            batch_size=16384,   # Minibatch size for SGD
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cuda"
        )

    # --- 5. Callbacks ---
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Single-file overwrite checkpoint: always writes to checkpoint_progress.zip
    checkpoint_callback = OverwriteCheckpointCallback(
        save_interval=args_cli.save_interval,
        save_path=checkpoint_dir,
        filename="checkpoint_progress",
        verbose=1,
    )

    # --- 6. Train ---
    print("Starting training...")
    model.learn(
        total_timesteps=args_cli.max_steps,
        callback=checkpoint_callback,
        reset_num_timesteps=False if args_cli.checkpoint else True
    )

    # Save final model separately so it's never overwritten
    final_path = os.path.join(checkpoint_dir, "yoink_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n====== TRAINING CRASHED — FULL TRACEBACK ======")
        traceback.print_exc()
        print("================================================\n")
        sys.exit(1)
