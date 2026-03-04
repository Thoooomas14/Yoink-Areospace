# Neural Network and PPO agent implementation

import torch
import torch.nn as nn
# Remove Isaac Lab env imports to prevent omni.physics initialization errors

from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 28 # 2 (pos) + 2 (target) + 24 (lidar)
        self.output_dim = 2
        
        # Policy Network (Actor)
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
            nn.Tanh() # Outputs -1.0 to 1.0
        )
        
        # Action standard deviation (learnable parameter)
        # Start with standard deviation of 1.0 (log_std = 0)
        self.action_log_std = nn.Parameter(torch.zeros(1, self.output_dim))
        
        # Value Network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a single value estimate
        )

    def forward(self, obs_tensor):
        action_mean = self.actor(obs_tensor)
        value = self.critic(obs_tensor)
        return action_mean, value
        
    def sample_action(self, obs_tensor):
        action_mean, value = self(obs_tensor)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        
        # Enforce action limits due to Tanh (-1, 1) environment assumptions
        action = torch.clamp(action, -1.0, 1.0)
        
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action, action_log_prob, value
        
    def evaluate(self, obs_tensor, action):
        action_mean, value = self(obs_tensor)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        dist = Normal(action_mean, action_std)
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        return action_log_prob, value, dist_entropy

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic().to("cuda")
        self.policy_old = ActorCritic().to("cuda")
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            action, action_log_prob, state_val = self.policy_old.sample_action(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(action_log_prob)
        self.buffer.values.append(state_val)

        return action
        
    def update(self):
        # Flatten the buffer tensors
        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.cat(self.buffer.actions, dim=0).detach()
        old_log_probs = torch.cat(self.buffer.log_probs, dim=0).detach()
        old_values = torch.cat(self.buffer.values, dim=0).detach()
        
        rewards = []
        discounted_reward = 0
        
        # Calculate returns
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if type(discounted_reward) is int:
                discounted_reward = torch.zeros_like(reward)
            
            # Reset discounted reward to 0 for environments that just terminated
            discounted_reward = torch.where(is_terminal, torch.zeros_like(discounted_reward), discounted_reward)
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.cat(rewards, dim=0).to("cuda")
        
        # Normalize returns
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Compute Advantages
        advantages = rewards.unsqueeze(1) - old_values
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Match state_values tensor dimensions with rewards tensor
            state_values = state_values.squeeze()
            
            # Core PPO ratio
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Clipped Surrogate Objective Loss
            surr1 = ratios * advantages.squeeze()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.squeeze()
            
            # Loss = -Clipping + Value_loss - Entropy_bonus
            # Minimize - (min(surr1, surr2)) which maximizes the surrogate objective
            v_loss = self.MseLoss(state_values, rewards.squeeze())
            loss = -torch.min(surr1, surr2) + 0.5 * v_loss - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()


