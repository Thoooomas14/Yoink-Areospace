import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
from stable_baselines3 import PPO

class YoinkInferenceNode(Node):
    def __init__(self):
        super().__init__('yoink_inference_node')
        
        # Load the trained model weights
        checkpoint_path = "/path/to/your/checkpoint_1.5B.zip"
        self.model = PPO.load(checkpoint_path, device="cuda") # Or "cpu" if no GPU on inference PC
        self.get_logger().info("SB3 Model Loaded Successfully")

        # Publisher: Send actions (v, omega) to the Raspberry Pi
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber: Receive combined sensor data from the Raspberry Pi
        self.sensor_sub = self.create_subscription(
            Float32MultiArray, 
            '/robot/observations', 
            self.sensor_callback, 
            10
        )

    def sensor_callback(self, msg):
        # 1. Reconstruct the observation array from incoming ROS message
        # Expected from Pi: [LinVel-X, YawRate, Goal-Dist, Goal-Angle, 24x LiDAR]
        obs = np.array(msg.data, dtype=np.float32)
        
        # Reshape for SB3 (requires batch dimension: (1, obs_dim))
        obs = obs.reshape(1, -1)
        
        # Clean the data exactly as you did in eval.py
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # 2. Run Inference
        actions, _ = self.model.predict(obs, deterministic=True)
        v = float(actions[0, 0])
        omega = float(actions[0, 1])

        # 3. Publish the Action
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        self.cmd_vel_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoinkInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()