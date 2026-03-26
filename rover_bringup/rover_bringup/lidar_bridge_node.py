import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.feature_publisher = self.create_publisher(
            Float32MultiArray,
            '/lidar_features',
            10
        )

        # 24 points: 0, 15, 30, ..., 345
        self.sample_angles_deg = [i * 15 for i in range(24)]

        self.get_logger().info(
            'Lidar node started. Sampling 24 points every 15 degrees.'
        )

    def get_range_at_angle(self, msg: LaserScan, angle_deg: float):
        angle_rad = math.radians(angle_deg)

        # Normalize to [-pi, pi)
        while angle_rad >= math.pi:
            angle_rad -= 2 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2 * math.pi

        if msg.angle_increment == 0.0:
            return None

        index = int(round((angle_rad - msg.angle_min) / msg.angle_increment))

        if index < 0 or index >= len(msg.ranges):
            return None

        value = msg.ranges[index]

        if math.isinf(value) or math.isnan(value):
            return None

        if value < msg.range_min or value > msg.range_max:
            return None

        return value

    def scan_callback(self, msg: LaserScan):
        sampled_ranges = []

        for angle in self.sample_angles_deg:
            r = self.get_range_at_angle(msg, angle)

            if r is None:
                r = msg.range_max

            sampled_ranges.append(float(r))

        feature_msg = Float32MultiArray()
        feature_msg.data = sampled_ranges
        self.feature_publisher.publish(feature_msg)

        self.get_logger().info(
            f'Published /lidar_features: {[round(x, 2) for x in sampled_ranges]}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()