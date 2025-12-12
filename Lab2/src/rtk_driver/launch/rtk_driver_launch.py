from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'port',
            default_value='/dev/pts/2',
            description='Serial port for RTK GPS device'
        ),
        
        DeclareLaunchArgument(
            'baud_rate',
            default_value='4800',
            description='Baud rate for serial communication'
        ),
        
        Node(
            package='rtk_driver',
            executable='rtk_driver',
            name='rtk_driver',
            output='screen',
            parameters=[{
                'port': LaunchConfiguration('port'),
                'baud_rate': LaunchConfiguration('baud_rate'),
            }]
        ),
    ])
