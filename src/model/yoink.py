import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# Note: We don't need to define ActorCritic or RolloutBuffer classes anymore.
# Stable Baselines3 handles these internally.

def get_ppo_agent(env, lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99):
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
    
    # Policy keyword arguments to match your previous architecture (128 -> 32)
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[128, 32], qf=[128, 32])
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