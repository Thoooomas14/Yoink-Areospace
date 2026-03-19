import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# Note: We don't need to define ActorCritic or RolloutBuffer classes anymore.
# Stable Baselines3 handles these internally.

def get_ppo_agent(env, lr=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99):
    """
    Initializes a Stable Baselines3 PPO agent.
    
    Args:
        env: The Isaac Lab environment instance (wrapped for SB3).
        lr: Learning rate.
        n_steps: The number of steps to run for each environment per update.
        batch_size: Minibatch size.
        n_epochs: Number of epochs when optimizing the surrogate loss.
        gamma: Discount factor.
    """
    
    # Policy keyword arguments to match your new architecture (256 -> 128)
    # Reduced log_std_init to -1.0 (std ~0.37) for more consistent exploration.
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        log_std_init=-1.0
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda"
    )
    
    return model