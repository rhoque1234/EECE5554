#!/usr/bin/env python3
"""
VectorNav VN-100 IMU Driver for ROS2
Reads VNYMR strings from serial port and publishes to /imu topic
"""

import rclpy
from rclpy.node import Node
import socket
import serial
import numpy as np
from custom_msg.msg import Vectornav
from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion, Vector3
import math


class VectorNavDriver(Node):
    def __init__(self):
        super().__init__('vn_driver_node')
        
        # Declare parameters
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('frame_id', 'imu1_frame')
        self.declare_parameter('use_tcp', True)
        self.declare_parameter('tcp_host', 'localhost')
        self.declare_parameter('tcp_port', 5555)
        
        # Get parameters
        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.use_tcp = self.get_parameter('use_tcp').value
        self.tcp_host = self.get_parameter('tcp_host').value
        self.tcp_port = self.get_parameter('tcp_port').value
        
        # Create publisher
        self.publisher = self.create_publisher(Vectornav, '/imu', 10)
        
        # Initialize connection
        self.connection = None
        self.connect()
        
        # Configure IMU to output at 40Hz
        self.configure_sample_rate()
        
        # Create timer to read data
        self.timer = self.create_timer(0.001, self.read_and_publish)  # Check frequently
        
        self.get_logger().info(f'VectorNav driver started. Publishing to /imu')
        self.get_logger().info(f'Frame ID: {self.frame_id}')
        
    def connect(self):
        """Connect to IMU"""
        try:
            if self.use_tcp:
                self.get_logger().info(f'Connecting to IMU at {self.tcp_host}:{self.tcp_port}...')
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect((self.tcp_host, self.tcp_port))
                self.connection.settimeout(0.1)
                self.get_logger().info('Connected to IMU')
            else:
                self.get_logger().info(f'Opening serial port {self.port} at {self.baudrate} baud...')
                self.connection = serial.Serial(self.port, self.baudrate, timeout=0.1)
                self.get_logger().info('Serial port opened')
        except Exception as e:
            self.get_logger().error(f'Failed to connect: {e}')
            raise
    
    def configure_sample_rate(self):
        """
        Configure VectorNav to output at 40Hz
        This writes to the async data output frequency register (Register 7)
        """
        try:
            # VectorNav register write command for 40Hz output
            # Format: $VNWRG,07,40*XX where XX is checksum
            command = "VNWRG,07,40"
            checksum = 0
            for char in command:
                checksum ^= ord(char)
            
            config_string = f"${command}*{checksum:02X}\r\n"
            
            if self.use_tcp:
                self.connection.sendall(config_string.encode('utf-8'))
            else:
                self.connection.write(config_string.encode('utf-8'))
            
            self.get_logger().info(f'Configured IMU to 40Hz: {config_string.strip()}')
        except Exception as e:
            self.get_logger().warn(f'Failed to configure sample rate: {e}')
    
    def read_line(self):
        """Read a line from the connection"""
        if self.use_tcp:
            # TCP socket reading
            buffer = b''
            while b'\n' not in buffer:
                try:
                    chunk = self.connection.recv(1)
                    if not chunk:
                        return None
                    buffer += chunk
                except socket.timeout:
                    return None
            return buffer.decode('utf-8', errors='ignore')
        else:
            # Serial port reading
            try:
                line = self.connection.readline().decode('utf-8', errors='ignore')
                return line if line else None
            except:
                return None
    
    def parse_vnymr(self, line):
        """
        Parse VNYMR string
        Format: $VNYMR,yaw,pitch,roll,mag_x,mag_y,mag_z,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z*checksum
        
        Returns: dict with parsed values or None if invalid
        """
        try:
            # Remove whitespace
            line = line.strip()
            
            # Check if valid VNYMR string
            if not line.startswith('$VNYMR'):
                return None
            
            # Split by * to separate data and checksum
            parts = line.split('*')
            if len(parts) != 2:
                return None
            
            data_part = parts[0]
            checksum_hex = parts[1]
            
            # Verify checksum
            calculated_checksum = 0
            for char in data_part[1:]:  # Skip the $
                calculated_checksum ^= ord(char)
            
            if calculated_checksum != int(checksum_hex, 16):
                self.get_logger().warn(f'Checksum mismatch: {line}')
                return None
            
            # Parse data fields
            fields = data_part.split(',')
            if len(fields) != 13:  # VNYMR + 12 data fields
                return None
            
            # Extract values (VectorNav units)
            parsed = {
                'yaw': float(fields[1]),      # degrees
                'pitch': float(fields[2]),    # degrees
                'roll': float(fields[3]),     # degrees
                'mag_x': float(fields[4]),    # Gauss
                'mag_y': float(fields[5]),    # Gauss
                'mag_z': float(fields[6]),    # Gauss
                'accel_x': float(fields[7]),  # m/s^2
                'accel_y': float(fields[8]),  # m/s^2
                'accel_z': float(fields[9]),  # m/s^2
                'gyro_x': float(fields[10]),  # deg/s
                'gyro_y': float(fields[11]),  # deg/s
                'gyro_z': float(fields[12])   # deg/s
            }
            
            return parsed
            
        except Exception as e:
            self.get_logger().warn(f'Failed to parse VNYMR string: {e}')
            return None
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion
        Args:
            roll, pitch, yaw: angles in degrees
        Returns:
            Quaternion (x, y, z, w)
        """
        # Convert to radians
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        
        # Calculate quaternion
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return (x, y, z, w)
    
    def read_and_publish(self):
        """Read IMU data and publish"""
        try:
            line = self.read_line()
            if line is None:
                return
            
            # Parse VNYMR string
            data = self.parse_vnymr(line)
            if data is None:
                return
            
            # Create timestamp
            now = self.get_clock().now().to_msg()
            
            # Create header
            header = Header()
            header.stamp = now
            header.frame_id = self.frame_id
            
            # Create IMU message
            imu_msg = Imu()
            imu_msg.header = header
            
            # Convert Euler angles to quaternion
            quat = self.euler_to_quaternion(data['roll'], data['pitch'], data['yaw'])
            imu_msg.orientation.x = quat[0]
            imu_msg.orientation.y = quat[1]
            imu_msg.orientation.z = quat[2]
            imu_msg.orientation.w = quat[3]
            
            # Convert gyro from deg/s to rad/s (ROS standard)
            imu_msg.angular_velocity.x = math.radians(data['gyro_x'])
            imu_msg.angular_velocity.y = math.radians(data['gyro_y'])
            imu_msg.angular_velocity.z = math.radians(data['gyro_z'])
            
            # Acceleration is already in m/s^2 (ROS standard)
            imu_msg.linear_acceleration.x = data['accel_x']
            imu_msg.linear_acceleration.y = data['accel_y']
            imu_msg.linear_acceleration.z = data['accel_z']
            
            # Set covariances (unknown, so set to -1)
            imu_msg.orientation_covariance = [-1.0] * 9
            imu_msg.angular_velocity_covariance = [-1.0] * 9
            imu_msg.linear_acceleration_covariance = [-1.0] * 9
            
            # Create MagneticField message
            mag_msg = MagneticField()
            mag_msg.header = header
            
            # Convert from Gauss to Tesla (1 Gauss = 1e-4 Tesla) - ROS standard
            mag_msg.magnetic_field.x = data['mag_x'] * 1e-4
            mag_msg.magnetic_field.y = data['mag_y'] * 1e-4
            mag_msg.magnetic_field.z = data['mag_z'] * 1e-4
            
            # Set covariance (unknown)
            mag_msg.magnetic_field_covariance = [-1.0] * 9
            
            # Create Vectornav message
            vn_msg = Vectornav()
            vn_msg.header = header
            vn_msg.imu = imu_msg
            vn_msg.mag_field = mag_msg
            vn_msg.raw_imu_string = line.strip()
            
            # Publish
            self.publisher.publish(vn_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in read_and_publish: {e}')
    
    def __del__(self):
        """Cleanup on shutdown"""
        if self.connection:
            try:
                if self.use_tcp:
                    self.connection.close()
                else:
                    self.connection.close()
            except:
                pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VectorNavDriver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
