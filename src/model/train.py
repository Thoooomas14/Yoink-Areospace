import sys
import os
import argparse
import traceback
import math
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
parser.add_argument("--tracking_weight",  type=float, default=0.6)
parser.add_argument("--motion_weight",  type=float, default=0.05)
parser.add_argument("--collision_weight", type=float, default=0.35)
parser.add_argument("--save_interval", type=int, default=25_000_000,
                    help="Save checkpoint every N timesteps")
parser.add_argument("--curriculum", action="store_true", help="Enable automatic reward weight shifting")
parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for PPO")
parser.add_argument("--reset_exploration", type=float, default=None, 
                    help="Reset log_std to this value when resuming (e.g., -0.7 for more noise)")

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
from src.model.yoink import get_ppo_agent


class CurriculumCallback(BaseCallback):
    """Gradually shifts reward weights over time.
    Phase 1 (Start): tracking=0.95, collision=0.05
    Phase 2 (End):   tracking=0.60, collision=0.40
    Interpolation happens between 'start_step' and 'end_step'.
    """

    def __init__(self, start_step: int, end_step: int, verbose: int = 1):
        super().__init__(verbose)
        self.start_step = start_step
        self.end_step = end_step

    def _on_step(self) -> bool:
        # Calculate interpolation factor [0, 1]
        t = (self.num_timesteps - self.start_step) / (self.end_step - self.start_step)
        t = max(0.0, min(1.0, t))

        # Linear interpolation for 3 weights (Total = 1.0)
        # Phase 1: tracking=0.80, motion=0.15, collision=0.05
        # Phase 2: tracking=0.60, motion=0.05 (keep active!), collision=0.35
        tracking_w  = 0.80 + t * (0.60 - 0.80)
        motion_w    = 0.15 + t * (0.05 - 0.15)
        collision_w = 0.05 + t * (0.35 - 0.05)

        try:
            # SB3 VecEnv unwrap logic:
            # We need to get to the ManagerBasedRLEnv to update _term_weights.
            # 1. try .unwrapped (standard gym/sb3)
            # 2. try .env (common for wrappers)
            curr_env = self.training_env
            while hasattr(curr_env, "env") or hasattr(curr_env, "envs"):
                if hasattr(curr_env, "envs"):
                    curr_env = curr_env.envs[0]
                else:
                    curr_env = curr_env.env
            
            isaac_env = curr_env
            rm = isaac_env.reward_manager
            
            # Use the official API to get term configs and update their weights
            rm.get_term_cfg("tracking").weight  = tracking_w
            rm.get_term_cfg("motion").weight    = motion_w
            rm.get_term_cfg("collision").weight = collision_w
            
            if self.num_timesteps % 100000 == 0 and self.verbose:
                print(f"[curriculum] Step {self.num_timesteps}: tracking={tracking_w:.3f}, motion={motion_w:.3f}, collision={collision_w:.3f}")
        except Exception as e:
            if self.num_timesteps % 100000 == 0:
                print(f"[curriculum] Warning: Could not update weights: {e}")

        return True


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


    # --- Terrain Diversity Calculation ---
    if args_cli.curriculum:
        # Phase 1 & 2 use a single terrain pattern for stability
        print("[train] Curriculum active: Using SINGLE terrain pattern for stability.")
        n_rows, n_cols = 1, 1
    else:
        # Aim for 128 robots per terrain patch
        num_terrains = max(1, args_cli.num_envs // 128)
        n_rows = int(math.sqrt(num_terrains))
        n_cols = num_terrains // n_rows
        print(f"[train] Configuring terrain grid: {n_rows}x{n_cols} ({num_terrains} unique patches)")
    
    env_cfg.scene.terrain.terrain_generator.num_rows = n_rows
    env_cfg.scene.terrain.terrain_generator.num_cols = n_cols

    # --- 4. Apply Weights to Config ---
    if args_cli.curriculum:
        # Start at Phase 1 weights: high tracking, high motion penalty, low collision
        print("[train] Initializing environment with Phase 1 weights (0.8/0.15/0.05)")
        env_cfg.rewards["tracking"].weight  = 0.80
        env_cfg.rewards["motion"].weight    = 0.15
        env_cfg.rewards["collision"].weight = 0.05
    else:
        # Use CLI arguments (defaults to Stage 2: 0.6/0.0/0.4)
        env_cfg.rewards["tracking"].weight  = args_cli.tracking_weight
        env_cfg.rewards["motion"].weight    = args_cli.motion_weight
        env_cfg.rewards["collision"].weight = args_cli.collision_weight

    # Create the environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # Wrap for SB3 compatibility (handles tensor conversion and normalization)
    env = Sb3VecEnvWrapper(env)

    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        print(f"Resuming training from: {args_cli.checkpoint}")
        model = PPO.load(args_cli.checkpoint, env=env, device="cuda")
        
        # Override entropy coefficient if provided
        if args_cli.ent_coef is not None:
            model.ent_coef = args_cli.ent_coef
            print(f"[train] Overriding ent_coef to {args_cli.ent_coef}")
            
        # Reset exploration (log_std) if requested
        if args_cli.reset_exploration is not None:
            with torch.no_grad():
                # For MlpPolicy, the log_std is a parameter of the policy
                model.policy.log_std.fill_(args_cli.reset_exploration)
            print(f"[train] Reset exploration log_std to {args_cli.reset_exploration}")
    else:
        print("Starting training from scratch.")
        model = get_ppo_agent(
            env,
            lr=3e-4,
            n_steps=1024,
            batch_size=32768,   
            n_epochs=10,        
            gamma=0.99,
            ent_coef=args_cli.ent_coef
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

    callbacks = [checkpoint_callback]
    if args_cli.curriculum:
        # Shift weights from 0.95/0.05 to 0.6/0.4 over 50M steps
        callbacks.append(CurriculumCallback(start_step=0, end_step=50_000_000))

    # --- 6. Train ---
    print("Starting training...")
    model.learn(
        total_timesteps=args_cli.max_steps,
        callback=callbacks,
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
