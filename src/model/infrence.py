import argparse
import math
import numpy as np

# --- 1. Launch Isaac Sim ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Digital Twin Inference for Yoink")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False  # Force visualization of digital twin
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
import isaaclab.utils.math as math_utils

# User environment imports
from isaac_env import LynxmotionSceneCfg

# ROS 2 and SB3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from stable_baselines3 import PPO

class YoinkInferenceNode(Node):
    def __init__(self):
        super().__init__('yoink_inference_node')
        
        # Load the trained model weights
        checkpoint_path = "/path/to/your/checkpoint.zip"
        self.model = PPO.load(checkpoint_path, device="cuda") # Or "cpu" if no GPU on inference PC
        self.get_logger().info("SB3 Model Loaded Successfully")

        # Publisher: Send actions (v, omega) to the Raspberry Pi
        self.cmd_vel_pub = self.create_publisher(Twist, '/policy_action', 10)

        # Subscriber: Receive combined sensor data from the Raspberry Pi
        self.sensor_sub = self.create_subscription(
            Float32MultiArray, 
            '/features', 
            self.sensor_callback, 
            10
        )
        
        # Telemetry Cache
        # Expected from Pi: [x, y, yaw, lin_vel_x, ang_vel_z, lidar(24)] = 29 values
        self.latest_telemetry = None
        self.new_data = False

    def sensor_callback(self, msg):
        self.latest_telemetry = np.array(msg.data, dtype=np.float32)
        self.new_data = True

def main():
    # 1. Setup digital twin scene
    scene_cfg = LynxmotionSceneCfg(num_envs=1, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim_utils.play_sim()

    # Get virtual target position from the Isaac environment
    target_pos_w = scene["target"].data.root_pos_w[0] # (x, y, z)
    target_x = target_pos_w[0].item()
    target_y = target_pos_w[1].item()

    # 2. Init ROS 2
    rclpy.init()
    node = YoinkInferenceNode()
    node.get_logger().info("Digital Twin Running. Waiting for Raspberry Pi telemetry on /robot/observations...")

    try:
        while simulation_app.is_running():
            # Process ROS callbacks (non-blocking)
            rclpy.spin_once(node, timeout_sec=0.01)
            
            # Require at least 29 values (5 telemetry + 24 lidar)
            if node.new_data and node.latest_telemetry is not None and len(node.latest_telemetry) >= 29:
                node.new_data = False
                data = node.latest_telemetry
                
                # Parse Telemetry
                x, y, yaw = data[0], data[1], data[2]
                lin_vel_x, ang_vel_z = data[3], data[4]
                lidar = data[5:29] # 24 physical rays
                
                # Update Digital Twin Pose in Isaac Sim
                robot = scene["robot"]
                robot_pos = torch.tensor([[x, y, 0.1]], device=scene.device)
                
                quaternion = math_utils.quat_from_euler_xyz(
                    torch.tensor([0.0], device=scene.device), 
                    torch.tensor([0.0], device=scene.device), 
                    torch.tensor([yaw], device=scene.device)
                )
                
                root_state = torch.cat([robot_pos, quaternion], dim=-1)
                robot.write_root_pose_to_sim(root_state)
                
                robot_vel = torch.tensor([[lin_vel_x, 0.0, 0.0, 0.0, 0.0, ang_vel_z]], device=scene.device)
                robot.write_root_velocity_to_sim(robot_vel)
                
                scene.write_data_to_sim()
                
                # Compute virtual target measurements
                target_vec_x = target_x - x
                target_vec_y = target_y - y
                r = math.sqrt(target_vec_x**2 + target_vec_y**2)
                
                target_yaw = math.atan2(target_vec_y, target_vec_x)
                theta = target_yaw - yaw
                theta = math.atan2(math.sin(theta), math.cos(theta)) # Wrap to [-pi, pi]
                
                r_norm = min(r / 10.0, 1.0)
                theta_norm = theta / math.pi
                
                # Construct Final Observation for the Model (Model specifically expects 28 dims: 2 IMU + 2 Target + 24 LiDAR)
                obs = np.zeros((1, 28), dtype=np.float32)
                
                # Velocities (from real IMU/odom)
                obs[0, 0] = np.clip(lin_vel_x / 1.5, -1.0, 1.0)
                obs[0, 1] = np.clip(ang_vel_z / 3.0, -1.0, 1.0)
                
                # Virtual Target Math
                obs[0, 2] = r_norm
                obs[0, 3] = theta_norm
                
                # Hardware LiDAR Array integration (24 rays: 0 to 345 degrees)
                lidar_24 = np.zeros(24, dtype=np.float32)
                lidar_len = min(len(lidar), 24)
                lidar_24[:lidar_len] = lidar[:lidar_len]

                obs[0, 4:] = np.clip(lidar_24 / 10.0, 0.0, 1.0)
                
                # Cleanup NaNs
                obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Run Inference
                actions, _ = node.model.predict(obs, deterministic=True)
                v = float(actions[0, 0])
                omega = float(actions[0, 1])

                # Scale actions to native real-world metrics mapping action parameters from training
                v_scaled = v * 2.0     # 2.0 m/s limit as per TwistActionCfg
                omega_scaled = omega * 3.14 # 3.14 rad/s limit as per TwistActionCfg

                # Publish Twist command
                twist_msg = Twist()
                twist_msg.linear.x = v_scaled
                twist_msg.angular.z = omega_scaled
                node.cmd_vel_pub.publish(twist_msg)

            # Keep simulation UI responsive and update sensor buffers (if any internal ones are running)
            scene.update(dt=0.01)
            simulation_app.update()

    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Digital Twin...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()

if __name__ == '__main__':
    main()