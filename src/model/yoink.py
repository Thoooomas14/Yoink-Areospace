import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from src.model.isaac_env import EnvCfg

class YoinkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 28 # 2 (pos) + 2 (target) + 24 (lidar)
        self.output_dim = 2
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh() # Outputs 0.0 to 1.0
        )

    def forward(self, obs_tensor):
        # The env will give you one big tensor of size [num_envs, 28]
        # It is already concatenated by the env observation manager
        return self.net(obs_tensor)
    

def main():
    # 1. Setup Env
    env_cfg = EnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 2. Setup Model
    model = YoinkModel().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    obs, _ = env.reset()
    current_obs = obs["policy"] # Shape: [num_envs, 28]

    for i in range(5000):
        # --- A. Get NN Actions (Shape: [N, 2]) ---
        # Output is (Left_Vel, Right_Vel) normalized 0-1
        nn_actions = model(current_obs) 

        # --- B. Map 2 Outputs to 4 Wheels ---
        # The env expects order: [FL, RL, FR, RR] based on the 'joint_names' list in EnvCfg
        # nn_actions[:, 0] is Left, nn_actions[:, 1] is Right
        
        left_cmd = nn_actions[:, 0].unsqueeze(1)  # Shape [N, 1]
        right_cmd = nn_actions[:, 1].unsqueeze(1) # Shape [N, 1]
        
        # Concatenate to make [N, 4]: (Left, Left, Right, Right)
        # We assume the joint order in EnvCfg was: FL, RL, FR, RR
        env_actions = torch.cat([left_cmd, left_cmd, right_cmd, right_cmd], dim=1)

        # --- C. Step Environment ---
        # Pass the expanded 4-wheel command to the environment
        obs, rewards, terms, truncs, infos = env.step({"wheels": env_actions})
        
        # --- D. Training Step ---
        loss = -rewards.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_obs = obs["policy"]

        if i % 100 == 0:
            print(f"Step {i} | Avg Reward: {rewards.mean().item():.4f}")
            
    env.close()

if __name__ == "__main__":
    main()
