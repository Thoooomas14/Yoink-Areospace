import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class SerialBridgeNode(Node):
    def __init__(self):
        super().__init__('serial_bridge_node')

        self.features_publisher = self.create_publisher(
            Float32MultiArray,
            '/features',
            10
        )

        self.lidar_subscription = self.create_subscription(
            Float32MultiArray,
            '/lidar_features',
            self.lidar_callback,
            10
        )

        self.action_subscription = self.create_subscription(
            Float32MultiArray,
            '/policy_action',
            self.action_callback,
            10
        )

        self.latest_lidar = None
        self.latest_serial_state = None

        self.serial_port = '/dev/ttyACM0'
        self.baud_rate = 115200

        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            self.get_logger().info(
                f'Opened serial bridge on {self.serial_port} at {self.baud_rate} baud'
            )
        except Exception as e:
            self.ser = None
            self.get_logger().error(f'Failed to open serial port: {e}')

        self.timer = self.create_timer(0.05, self.read_serial_data)

        self.get_logger().info(
            'Serial bridge node started. Reading serial state, combining with /lidar_features, and forwarding /policy_action.'
        )

    def lidar_callback(self, msg: Float32MultiArray):
        lidar_features = list(msg.data)

        if len(lidar_features) != 24:
            self.get_logger().warn(
                f'Expected 24 lidar values, got {len(lidar_features)}'
            )
            return

        self.latest_lidar = lidar_features
        self.publish_features()

    def read_serial_data(self):
        if self.ser is None:
            return

        try:
            if self.ser.in_waiting <= 0:
                return

            line = self.ser.readline().decode('utf-8').strip()

            if not line:
                return

            parts = line.split(',')

            if len(parts) != 5:
                self.get_logger().warn(f'Invalid serial data format: {line}')
                return

            x_position = float(parts[0])
            y_position = float(parts[1])
            heading = float(parts[2])
            linear_velocity = float(parts[3])
            angular_velocity = float(parts[4])
            self.latest_serial_state = [
                x_position,
                y_position,
                heading,
                linear_velocity,
                angular_velocity,
            ]
            self.publish_features()
        except Exception as e:
            self.get_logger().error(f'Error reading serial data: {e}')

    def publish_features(self):
        if self.latest_serial_state is None or self.latest_lidar is None:
            return

        combined_features = list(self.latest_serial_state) + list(self.latest_lidar)

        msg = Float32MultiArray()
        msg.data = [float(value) for value in combined_features]
        self.features_publisher.publish(msg)

        self.get_logger().info(
            f'Published /features with {len(combined_features)} values'
        )

    def action_callback(self, msg: Float32MultiArray):
        action = list(msg.data)

        if len(action) != 2:
            self.get_logger().warn(f'Expected 2 action values, got {len(action)}')
            return

        linear_velocity = float(action[0])
        angular_velocity = float(action[1])
        self.send_to_serial(linear_velocity, angular_velocity)

    def send_to_serial(self, linear_velocity, angular_velocity):
        if self.ser is None:
            self.get_logger().error('Serial connection is not available')
            return

        command = f'{linear_velocity:.3f},{angular_velocity:.3f}\n'

        try:
            self.ser.write(command.encode('utf-8'))
            self.get_logger().info(f'Sent to serial: {command.strip()}')
        except Exception as e:
            self.get_logger().error(f'Failed to send serial command: {e}')

    def destroy_node(self):
        if hasattr(self, 'ser') and self.ser is not None:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
