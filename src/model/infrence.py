import os
import argparse
import math
import numpy as np
import traceback

# --- 1. Launch Isaac Sim ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Digital Twin Inference for Yoink")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--domain_id", type=int, default=15, help="ROS 2 Domain ID")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Match the ROS 2 Domain ID of the rover (standardized to 15 for your setup)
os.environ["ROS_DOMAIN_ID"] = str(args_cli.domain_id)
args_cli.headless = False  # Force visualization of digital twin
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
import isaaclab.utils.math as math_utils

# User environment imports
from isaac_env import LynxmotionSceneCfg, HfRandomUniformTerrainCfg
import omni.ui as ui

# ROS 2 and SB3
import omni.kit.app
manager = omni.kit.app.get_app().get_extension_manager()

# Enable the modern ROS 2 bridge extension
manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

def build_ui():
    """Build the debug UI window (Ported from eval.py)."""
    window = ui.Window("Lynxmotion SB3 Digital Twin", width=450, height=900)
    labels = {}
    with window.frame:
        with ui.VStack(spacing=8, style={"padding": 14}):
            ui.Label("LIVE SB3 NEURAL NETWORK I/O",
                     style={"font_size": 20, "color": 0xFF00FF00})
            ui.Line(style={"color": 0xFF555555})
            ui.Label("--- ROS OBSERVATIONS ---",
                     style={"font_size": 13, "color": 0xFF00AAFF})
            labels["imu"]    = ui.Label("IMU  [LinVel-X, YawRate]: ---",
                                        style={"font_size": 13, "color": 0xFFFFFFFF})
            labels["target"] = ui.Label("Goal [Dist-norm, Angle]:  ---",
                                        style={"font_size": 13, "color": 0xFFFFFFFF})
            # Radar Display (Stabilized Local Version)
            ui.Spacer(height=40)
            ui.Label("--- LIDAR RADAR (Real-World) ---",
                     style={"font_size": 13, "color": 0xFF00AAFF})
            
            with ui.HStack(height=260):
                ui.Spacer()
                with ui.ZStack(width=260, height=260):
                    # 1. Background Disk
                    with ui.Placer(offset_x=10, offset_y=10):
                        ui.Circle(width=240, height=240, style={"background_color": 0xFF080808})
                    
                    # 2. Scale Rings and Crosshairs
                    with ui.Placer(offset_x=10, offset_y=10):
                        ui.Circle(width=240, height=240, style={"color": 0x44AAAAAA, "thickness": 1})
                    with ui.Placer(offset_x=70, offset_y=70):
                        ui.Circle(width=120, height=120, style={"color": 0x22888888, "thickness": 1})
                    
                    with ui.Placer(offset_x=10, offset_y=130):
                        ui.Line(width=240, height=0, style={"color": 0x33888888})
                    with ui.Placer(offset_x=130, offset_y=10):
                        ui.Line(width=0, height=240, style={"color": 0x33888888})
                    
                    # 3. Lidar Detection Boundary (Stable Polyline)
                    try:
                        labels["radar_poly"] = ui.Polyline(points=[(130,130)]*2, 
                                                            style={"color": 0x4400FF00, "thickness": 1})
                    except:
                        labels["radar_poly"] = None

                    # 4. Lidar Dots
                    radar_dots = []
                    for i in range(24):
                        dp = ui.Placer(offset_x=127, offset_y=127)
                        dp.visible = False
                        with dp:
                            ui.Circle(width=6, height=6, style={"background_color": 0xFF00FF00})
                        radar_dots.append(dp)
                    
                    # 5. Robot Center (Facing Arrow)
                    with ui.Placer(offset_x=128, offset_y=120):
                        ui.Rectangle(width=4, height=10, style={"background_color": 0xFFAAAA00})

                    labels["radar_dots"] = radar_dots
                ui.Spacer()
            ui.Spacer(height=40)

            ui.Line(style={"color": 0xFF555555})
            ui.Label("--- ACTIONS / POLICY OUTPUT ---",
                     style={"font_size": 13, "color": 0xFFFFCC00})
            labels["action"] = ui.Label("Twist [v, omega]: ---",
                                        style={"font_size": 16, "color": 0xFFFFFFFF})
            labels["action_raw"] = ui.Label("Policy Raw Output: [---, ---]",
                                         style={"font_size": 11, "color": 0xFFAAAA00})

    return window, labels
# (If it complains it can't find that name, use the legacy name: "omni.isaac.ros2_bridge")

# Force the engine to process the extension load and refresh paths
simulation_app.update()
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from stable_baselines3 import PPO

class YoinkInferenceNode(Node):
    def __init__(self):
        super().__init__('yoink_inference_node')
        
        # Load the trained model weights
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "checkpoint_600M.zip")
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
    # 0. Setup simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    # 1. Setup digital twin scene
    scene_cfg = LynxmotionSceneCfg(num_envs=1, env_spacing=0.0)
    # Remove procedural obstacles for the digital twin
    # Switching to a flat, obstacle-free floor since physical LiDAR picks up real-world obstacles.
    scene_cfg.terrain.terrain_generator.sub_terrains = {
        "flat": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.0, 0.0), noise_step=1.0)
    }
    scene_cfg.lidar.debug_vis = False
    scene = InteractiveScene(scene_cfg)

    # Initialize Debug UI
    debug_window, lbl = build_ui()
    
    # 2. Start the simulation
    sim.reset()
    simulation_app.update()

    # Get virtual target position from the Isaac environment
    target_pos_w = scene["target"].data.root_pos_w[0] # (x, y, z)
    target_x = target_pos_w[0].item()
    target_y = target_pos_w[1].item()

    # 2. Init ROS 2
    rclpy.init()
    node = YoinkInferenceNode()
    node.get_logger().info("Digital Twin Running. Waiting for Raspberry Pi telemetry on /features...")

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

                # Update UI labels
                lbl["imu"].text = f"IMU  [LinVel-X, YawRate]:  {obs[0,0]:+.3f}  {obs[0,1]:+.3f}"
                lbl["target"].text = f"Goal [Dist-norm, Angle]:   {obs[0,2]:+.3f}  {obs[0,3]:+.3f}"
                lbl["action"].text = f"Twist [v={v_scaled:+.2f}   omega={omega_scaled:+.2f}]"
                lbl["action_raw"].text = f"Policy [v_raw={v:+.3f}   w_raw={omega:+.3f}]"

                # Update Radar Display (ROS LiDAR 24 rays mapped CW/CCW as per eval.py logic)
                rays = obs[0, 4:] # The 24 normalized LiDAR rays
                poly_points = []
                for i in range(24):
                    # Native CCW mapping for display
                    angle_rad = math.radians(i * (360.0 / 24.0))
                    dist_norm = float(rays[i])
                    r_ui = dist_norm * 110.0 # Scale to radar radius
                    
                    # CCW math: dx = -sin, dy = -cos (0 = UP)
                    dx = -math.sin(angle_rad) * r_ui
                    dy = -math.cos(angle_rad) * r_ui
                    
                    lbl["radar_dots"][i].offset_x = 127 + dx
                    lbl["radar_dots"][i].offset_y = 127 + dy
                    lbl["radar_dots"][i].visible = True
                    poly_points.append((130 + dx, 130 + dy))

                if poly_points and lbl["radar_poly"]:
                    poly_points.append(poly_points[0]) # close loop
                    lbl["radar_poly"].points = poly_points

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