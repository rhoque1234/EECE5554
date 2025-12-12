#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import utm
from datetime import datetime


class RTKDriver(Node):
    def __init__(self):
        super().__init__('rtk_driver')
        
        # Declare parameters
        self.declare_parameter('port', '/dev/pts/2')
        self.declare_parameter('baud_rate', 4800)
        
        # Get parameters
        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        
        # Create publisher - using String for now, will use Customrtk.msg in final version
        self.publisher_ = self.create_publisher(String, 'rtk_data', 10)
        
        # Initialize serial connection
        try:
            self.serial_port = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.get_logger().info(f'Connected to {self.port} at {self.baud_rate} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            return
        
        # Create timer to read data
        self.timer = self.create_timer(0.1, self.read_and_publish)
        
    def parse_gngga(self, sentence):
        """Parse GNGGA sentence and extract relevant data"""
        try:
            parts = sentence.strip().split(',')
            
            if not sentence.startswith('$GNGGA') or len(parts) < 15:
                return None
            
            # Extract UTC time
            utc_time = parts[1]
            
            # Parse latitude
            lat_str = parts[2]
            lat_dir = parts[3]
            if lat_str and lat_dir:
                lat_deg = float(lat_str[:2])
                lat_min = float(lat_str[2:])
                latitude = lat_deg + lat_min / 60.0
                if lat_dir == 'S':
                    latitude = -latitude
            else:
                return None
            
            # Parse longitude
            lon_str = parts[4]
            lon_dir = parts[5]
            if lon_str and lon_dir:
                lon_deg = float(lon_str[:3])
                lon_min = float(lon_str[3:])
                longitude = lon_deg + lon_min / 60.0
                if lon_dir == 'W':
                    longitude = -longitude
            else:
                return None
            
            # Fix quality (0=invalid, 1=GPS fix, 2=DGPS fix, 4=RTK fixed, 5=RTK float)
            fix_quality = int(parts[6]) if parts[6] else 0
            
            # Number of satellites
            num_satellites = int(parts[7]) if parts[7] else 0
            
            # HDOP
            hdop = float(parts[8]) if parts[8] else 999.9
            
            # Altitude
            altitude = float(parts[9]) if parts[9] else 0.0
            
            # Convert to UTM
            utm_data = utm.from_latlon(latitude, longitude)
            easting = utm_data[0]
            northing = utm_data[1]
            zone_num = utm_data[2]
            zone_letter = utm_data[3]
            
            return {
                'utc_time': utc_time,
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude,
                'fix_quality': fix_quality,
                'num_satellites': num_satellites,
                'hdop': hdop,
                'easting': easting,
                'northing': northing,
                'zone_num': zone_num,
                'zone_letter': zone_letter,
                'gngga_string': sentence.strip()
            }
            
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'Error parsing GNGGA: {e}')
            return None
    
    def read_and_publish(self):
        """Read data from serial port and publish"""
        try:
            if self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode('utf-8', errors='ignore')
                
                if line.startswith('$GNGGA'):
                    parsed_data = self.parse_gngga(line)
                    
                    if parsed_data:
                        # Create message (using String for now)
                        msg = String()
                        msg.data = (f"UTC: {parsed_data['utc_time']}, "
                                   f"Lat: {parsed_data['latitude']:.7f}, "
                                   f"Lon: {parsed_data['longitude']:.7f}, "
                                   f"Alt: {parsed_data['altitude']:.2f}, "
                                   f"E: {parsed_data['easting']:.2f}, "
                                   f"N: {parsed_data['northing']:.2f}, "
                                   f"Zone: {parsed_data['zone_num']}{parsed_data['zone_letter']}, "
                                   f"Fix: {parsed_data['fix_quality']}, "
                                   f"HDOP: {parsed_data['hdop']:.1f}, "
                                   f"Sats: {parsed_data['num_satellites']}")
                        
                        self.publisher_.publish(msg)
                        self.get_logger().info(f"Published: Fix={parsed_data['fix_quality']}, HDOP={parsed_data['hdop']:.1f}")
                        
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')
    
    def __del__(self):
        """Clean up serial connection"""
        if hasattr(self, 'serial_port') and self.serial_port.is_open:
            self.serial_port.close()


def main(args=None):
    rclpy.init(args=args)
    rtk_driver = RTKDriver()
    
    try:
        rclpy.spin(rtk_driver)
    except KeyboardInterrupt:
        pass
    finally:
        rtk_driver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
