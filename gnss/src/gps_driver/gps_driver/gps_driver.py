import rclpy
from rclpy.node import Node
from gps_driver.msg import Customgps
import serial
import utm
from datetime import datetime, timezone
import time

class GPSDriver(Node):
    def __init__(self):
        super().__init__('gps_driver')
        
        # Declare and get the port parameter
        self.declare_parameter('port', '/dev/ttyUSB0')
        port = self.get_parameter('port').get_parameter_value().string_value
        
        self.get_logger().info(f'Opening serial port: {port}')
        
        # Open serial connection
        try:
            self.ser = serial.Serial(port, baudrate=4800, timeout=1)
            self.get_logger().info('Serial port opened successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise
        
        # Create publisher
        self.publisher_ = self.create_publisher(Customgps, '/gps', 10)
        
        # Create timer to read GPS data
        self.timer = self.create_timer(0.1, self.timer_callback)
        
    def parse_gpgga(self, gpgga_string):
        """Parse GPGGA string and return dictionary of values"""
        parts = gpgga_string.split(',')
        
        if len(parts) < 15 or parts[0] != '$GPGGA':
            return None
        
        try:
            # Parse UTC time
            utc_str = parts[1]
            if not utc_str:
                return None
            
            # Parse latitude
            lat_str = parts[2]
            lat_dir = parts[3]
            if not lat_str or not lat_dir:
                return None
            
            lat_deg = float(lat_str[:2])
            lat_min = float(lat_str[2:])
            latitude = lat_deg + lat_min / 60.0
            if lat_dir == 'S':
                latitude = -latitude
            
            # Parse longitude
            lon_str = parts[4]
            lon_dir = parts[5]
            if not lon_str or not lon_dir:
                return None
            
            lon_deg = float(lon_str[:3])
            lon_min = float(lon_str[3:])
            longitude = lon_deg + lon_min / 60.0
            if lon_dir == 'W':
                longitude = -longitude
            
            # Parse altitude
            altitude = float(parts[9]) if parts[9] else 0.0
            
            # Parse HDOP
            hdop = float(parts[8]) if parts[8] else 0.0
            
            # Convert UTC time to epoch
            hours = int(utc_str[0:2])
            minutes = int(utc_str[2:4])
            seconds = int(utc_str[4:6])
            
            # Get current date for epoch conversion
            now = datetime.now(timezone.utc)
            utc_time = datetime(now.year, now.month, now.day, 
                              hours, minutes, seconds, tzinfo=timezone.utc)
            epoch_time = utc_time.timestamp()
            
            sec = int(epoch_time)
            nsec = int((epoch_time - sec) * 1e9)
            
            # Convert to UTM
            utm_coords = utm.from_latlon(latitude, longitude)
            
            return {
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude,
                'utm_easting': utm_coords[0],
                'utm_northing': utm_coords[1],
                'zone': utm_coords[2],
                'letter': utm_coords[3],
                'hdop': hdop,
                'sec': sec,
                'nsec': nsec
            }
            
        except Exception as e:
            self.get_logger().error(f'Error parsing GPGGA: {e}')
            return None
    
    def timer_callback(self):
        """Read and publish GPS data"""
        try:
            # Read line from serial port
            line = self.ser.readline().decode('ascii', errors='ignore').strip()
            
            if line.startswith('$GPGGA'):
                # Parse the GPGGA string
                parsed_data = self.parse_gpgga(line)
                
                if parsed_data:
                    # Create and populate message
                    msg = Customgps()
                    msg.header.frame_id = 'GPS1_Frame'
                    msg.header.stamp.sec = parsed_data['sec']
                    msg.header.stamp.nanosec = parsed_data['nsec']
                    msg.latitude = parsed_data['latitude']
                    msg.longitude = parsed_data['longitude']
                    msg.altitude = parsed_data['altitude']
                    msg.utm_easting = parsed_data['utm_easting']
                    msg.utm_northing = parsed_data['utm_northing']
                    msg.zone = parsed_data['zone']
                    msg.letter = parsed_data['letter']
                    msg.hdop = parsed_data['hdop']
                    msg.gpgga_read = line
                    
                    # Publish message
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'Published GPS data: Lat={msg.latitude:.6f}, Lon={msg.longitude:.6f}')
                    
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')
    
    def destroy_node(self):
        """Clean up serial port on shutdown"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        gps_driver = GPSDriver()
        rclpy.spin(gps_driver)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()