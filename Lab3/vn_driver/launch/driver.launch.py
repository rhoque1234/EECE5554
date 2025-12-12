#!/usr/bin/env python3
"""
Launch file for VectorNav VN-100 IMU driver
Usage: ros2 launch vn_driver driver.launch.py port:=/dev/ttyUSB0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyUSB0',
        description='Serial port for VectorNav IMU'
    )
    
    use_tcp_arg = DeclareLaunchArgument(
        'use_tcp',
        default_value='true',
        description='Use TCP connection instead of serial port'
    )
    
    tcp_host_arg = DeclareLaunchArgument(
        'tcp_host',
        default_value='localhost',
        description='TCP host address'
    )
    
    tcp_port_arg = DeclareLaunchArgument(
        'tcp_port',
        default_value='5555',
        description='TCP port'
    )
    
    # Create node
    vn_driver_node = Node(
        package='vn_driver',
        executable='vn_driver_node',
        name='vn_driver',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('port'),
            'baudrate': 115200,
            'frame_id': 'imu1_frame',
            'use_tcp': LaunchConfiguration('use_tcp'),
            'tcp_host': LaunchConfiguration('tcp_host'),
            'tcp_port': LaunchConfiguration('tcp_port'),
        }]
    )
    
    return LaunchDescription([
        port_arg,
        use_tcp_arg,
        tcp_host_arg,
        tcp_port_arg,
        vn_driver_node
    ])
