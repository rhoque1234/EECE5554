from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'port',
            default_value='/dev/ttyACM0',
            description='Serial port for GPS device'
        ),
        
        Node(
            package='gps_driver',
            executable='gps_driver',
            name='gps_driver',
            parameters=[{
                'port': LaunchConfiguration('port')
            }],
            output='screen'
        )
    ])