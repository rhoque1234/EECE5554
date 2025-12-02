from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file for GPS and IMU drivers.
    This launch file starts both the GPS and IMU driver nodes simultaneously
    for data collection during Lab 5.
    """
    
    return LaunchDescription([
        # GPS Driver Node
        Node(
            package='gps_driver',
            executable='gps_driver_node',
            name='gps_driver',
            output='screen',
            parameters=[{
                'port': '/dev/ttyUSB0',  # Adjust based on your GPS device
                'baud_rate': 4800,
            }]
        ),
        
        # IMU Driver Node
        Node(
            package='imu_driver',
            executable='imu_driver_node',
            name='imu_driver',
            output='screen',
            parameters=[{
                'port': '/dev/ttyUSB1',  # Adjust based on your IMU device
                'baud_rate': 115200,
            }]
        ),
    ])
