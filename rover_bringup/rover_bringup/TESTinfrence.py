import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class YoinkInferenceNode(Node):
    EXPECTED_FEATURE_COUNT = 29
    LIDAR_SAMPLE_COUNT = 24
    MAX_LIDAR_RANGE = 10.0

    STOP_DISTANCE_M = 0.45
    SLOW_DISTANCE_M = 0.80
    CRUISE_LINEAR_VEL = 0.35
    SLOW_LINEAR_VEL = 0.15
    FAST_TURN_RATE = 1.00
    SLOW_TURN_RATE = 0.55
    OPEN_SPACE_TURN_GAIN = 0.35

    def __init__(self):
        super().__init__('yoink_inference_node')

        self.cmd_vel_pub = self.create_publisher(Twist, '/policy_action', 10)
        self.sensor_sub = self.create_subscription(
            Float32MultiArray,
            '/features',
            self.sensor_callback,
            10,
        )

        self.last_turn_direction = 1.0

        self.get_logger().info(
            'Hardcoded inference node running. Listening on /features and publishing /policy_action.'
        )

    def sensor_callback(self, msg: Float32MultiArray):
        if len(msg.data) < self.EXPECTED_FEATURE_COUNT:
            self.get_logger().warn(
                f'Expected at least {self.EXPECTED_FEATURE_COUNT} features, got {len(msg.data)}'
            )
            self.publish_command(0.0, 0.0)
            return

        data = [float(value) for value in msg.data[:self.EXPECTED_FEATURE_COUNT]]

        x_position = data[0]
        y_position = data[1]
        yaw = data[2]
        linear_velocity = data[3]
        angular_velocity = data[4]
        lidar = self.sanitize_lidar(data[5:5 + self.LIDAR_SAMPLE_COUNT])

        linear_cmd, angular_cmd = self.compute_command(
            x_position,
            y_position,
            yaw,
            linear_velocity,
            angular_velocity,
            lidar,
        )
        self.publish_command(linear_cmd, angular_cmd)

        self.get_logger().info(
            f'cmd lin={linear_cmd:.3f} ang={angular_cmd:.3f} | '
            f'pose=({x_position:.2f}, {y_position:.2f}, {yaw:.2f}) | '
            f'vel=({linear_velocity:.2f}, {angular_velocity:.2f})'
        )

    def sanitize_lidar(self, lidar_values):
        sanitized = []
        for value in lidar_values:
            if not math.isfinite(value) or value <= 0.0:
                sanitized.append(self.MAX_LIDAR_RANGE)
            else:
                sanitized.append(min(value, self.MAX_LIDAR_RANGE))

        if len(sanitized) < self.LIDAR_SAMPLE_COUNT:
            sanitized.extend(
                [self.MAX_LIDAR_RANGE] * (self.LIDAR_SAMPLE_COUNT - len(sanitized))
            )

        return sanitized[:self.LIDAR_SAMPLE_COUNT]

    def sector_min(self, lidar_values, indices):
        return min(lidar_values[index] for index in indices)

    def choose_turn_direction(self, left_clearance, right_clearance):
        clearance_delta = left_clearance - right_clearance

        if clearance_delta > 0.05:
            self.last_turn_direction = 1.0
        elif clearance_delta < -0.05:
            self.last_turn_direction = -1.0

        return self.last_turn_direction

    def compute_command(
        self,
        x_position,
        y_position,
        yaw,
        linear_velocity,
        angular_velocity,
        lidar,
    ):
        del x_position, y_position, yaw, linear_velocity, angular_velocity

        # Lidar samples are ordered at 0, 15, 30, ..., 345 degrees.
        front_min = self.sector_min(lidar, [23, 0, 1])
        front_left_min = self.sector_min(lidar, [1, 2, 3, 4])
        front_right_min = self.sector_min(lidar, [20, 21, 22, 23])
        left_min = self.sector_min(lidar, [4, 5, 6, 7, 8])
        right_min = self.sector_min(lidar, [16, 17, 18, 19, 20])

        turn_direction = self.choose_turn_direction(left_min, right_min)

        if (
            front_min < self.STOP_DISTANCE_M
            or front_left_min < self.STOP_DISTANCE_M
            or front_right_min < self.STOP_DISTANCE_M
        ):
            return 0.0, self.FAST_TURN_RATE * turn_direction

        if front_min < self.SLOW_DISTANCE_M:
            return self.SLOW_LINEAR_VEL, self.SLOW_TURN_RATE * turn_direction

        open_space_bias = (left_min - right_min) * self.OPEN_SPACE_TURN_GAIN
        open_space_bias = max(-0.35, min(0.35, open_space_bias))

        if abs(open_space_bias) < 0.05:
            open_space_bias = 0.0

        return self.CRUISE_LINEAR_VEL, open_space_bias

    def publish_command(self, linear_velocity, angular_velocity):
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_velocity)
        twist_msg.angular.z = float(angular_velocity)
        self.cmd_vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoinkInferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down hardcoded inference node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
