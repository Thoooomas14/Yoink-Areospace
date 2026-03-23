from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scan_topic = LaunchConfiguration('scan_topic')
    start_rplidar = LaunchConfiguration('start_rplidar')

    return LaunchDescription([
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='LaserScan topic consumed by lidar_bridge_node.'
        ),
        DeclareLaunchArgument(
            'start_rplidar',
            default_value='false',
            description='If true, also launch the rplidar_ros driver package.'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('rplidar_ros'),
                    'launch',
                    'rplidar.launch.py',
                ])
            ),
            condition=IfCondition(start_rplidar),
        ),

        Node(
            package='rover_bringup',
            executable='lidar_bridge_node',
            name='lidar_bridge_node',
            remappings=[
                ('/scan', scan_topic),
            ],
            output='screen'
        ),

        Node(
            package='rover_bringup',
            executable='serial_bridge_node',
            name='serial_bridge_node',
            output='screen'
        ),

        Node(
            package='rover_bringup',
            executable='model_subscriber',
            name='model_subscriber',
            output='screen'
        ),
    ])
