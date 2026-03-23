from setuptools import setup

package_name = 'rover_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/rover.launch.py']),
    ],
    install_requires=[
        'setuptools',
        'pyserial',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='ROS 2 bringup package for Yoink rover integration.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lidar_bridge_node = rover_bringup.lidar_bridge_node:main',
            'model_subscriber = rover_bringup.model_subscriber:main',
            'serial_bridge_node = rover_bringup.serial_bridge_node:main',
        ],
    },
)
