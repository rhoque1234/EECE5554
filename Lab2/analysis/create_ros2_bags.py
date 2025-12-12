#!/usr/bin/env python3
"""
Create ROS 2 bag files from RTK .txt data without requiring ROS 2 installation.
This creates SQLite3-based bag files compatible with ROS 2 Humble/Iron/Jazzy.
"""

import sqlite3
import struct
import time
import yaml
import utm
from pathlib import Path
from datetime import datetime


class ROS2BagWriter:
    """Minimal ROS 2 bag writer that creates SQLite3 bag files"""
    
    def __init__(self, bag_path):
        self.bag_path = Path(bag_path)
        self.bag_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata.yaml
        self.create_metadata()
        
        # Create SQLite database
        self.db_path = self.bag_path / 'lab2_master_0.db3'
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self.create_tables()
        
        self.topic_id = None
        self.message_count = 0
        
    def create_metadata(self):
        """Create ROS 2 bag metadata.yaml file"""
        metadata = {
            'rosbag2_bagfile_information': {
                'version': 5,
                'storage_identifier': 'sqlite3',
                'relative_file_paths': ['lab2_master_0.db3'],
                'duration': {
                    'nanoseconds': 0
                },
                'starting_time': {
                    'nanoseconds_since_epoch': int(time.time() * 1e9)
                },
                'message_count': 0,
                'topics_with_message_count': [],
                'compression_format': '',
                'compression_mode': ''
            }
        }
        
        with open(self.bag_path / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    
    def create_tables(self):
        """Create ROS 2 bag database tables"""
        # Topics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                serialization_format TEXT NOT NULL,
                offered_qos_profiles TEXT
            )
        ''')
        
        # Messages table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL
            )
        ''')
        
        self.conn.commit()
    
    def create_topic(self, topic_name, message_type):
        """Register a topic in the bag"""
        self.cursor.execute(
            'INSERT INTO topics (name, type, serialization_format, offered_qos_profiles) VALUES (?, ?, ?, ?)',
            (topic_name, message_type, 'cdr', '')
        )
        self.conn.commit()
        self.topic_id = self.cursor.lastrowid
        return self.topic_id
    
    def serialize_string_message(self, data_dict):
        """
        Serialize a custom RTK message to CDR format.
        This creates a simplified serialization that includes all RTK fields.
        """
        # CDR header (4 bytes): encapsulation kind + flags
        cdr_header = struct.pack('<HH', 0, 1)  # Little endian, version 1
        
        # Create a simple string representation of the message
        msg_str = (
            f"UTC: {data_dict['utc_time']}, "
            f"Lat: {data_dict['latitude']:.8f}, "
            f"Lon: {data_dict['longitude']:.8f}, "
            f"Alt: {data_dict['altitude']:.2f}, "
            f"E: {data_dict['easting']:.2f}, "
            f"N: {data_dict['northing']:.2f}, "
            f"Zone: {data_dict['zone_num']}{data_dict['zone_letter']}, "
            f"Fix: {data_dict['fix_quality']}, "
            f"HDOP: {data_dict['hdop']:.2f}, "
            f"Sats: {data_dict['num_sats']}"
        )
        
        # String length (4 bytes) + string data + null terminator
        msg_bytes = msg_str.encode('utf-8')
        str_length = struct.pack('<I', len(msg_bytes) + 1)
        
        return cdr_header + str_length + msg_bytes + b'\x00'
    
    def write_message(self, data_dict, timestamp_ns):
        """Write a message to the bag"""
        serialized_data = self.serialize_string_message(data_dict)
        
        self.cursor.execute(
            'INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)',
            (self.topic_id, timestamp_ns, serialized_data)
        )
        self.message_count += 1
    
    def update_metadata(self, start_time_ns, end_time_ns):
        """Update metadata with final statistics"""
        duration_ns = end_time_ns - start_time_ns
        
        metadata = {
            'rosbag2_bagfile_information': {
                'version': 5,
                'storage_identifier': 'sqlite3',
                'relative_file_paths': ['lab2_master_0.db3'],
                'duration': {
                    'nanoseconds': int(duration_ns)
                },
                'starting_time': {
                    'nanoseconds_since_epoch': int(start_time_ns)
                },
                'message_count': self.message_count,
                'topics_with_message_count': [
                    {
                        'topic_metadata': {
                            'name': '/rtk_data',
                            'type': 'std_msgs/msg/String',
                            'serialization_format': 'cdr',
                            'offered_qos_profiles': ''
                        },
                        'message_count': self.message_count
                    }
                ],
                'compression_format': '',
                'compression_mode': ''
            }
        }
        
        with open(self.bag_path / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    
    def close(self):
        """Close the bag file"""
        self.conn.commit()
        self.conn.close()


def parse_gngga(line):
    """Parse GNGGA sentence and extract key data"""
    if not line.startswith('$GNGGA'):
        return None
    
    parts = line.strip().split(',')
    if len(parts) < 15:
        return None
    
    try:
        # UTC time
        utc_time = parts[1]
        
        # Latitude
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
        
        # Longitude
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
        
        # Fix quality
        fix_quality = int(parts[6]) if parts[6] else 0
        
        # Number of satellites
        num_sats = int(parts[7]) if parts[7] else 0
        
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
            'num_sats': num_sats,
            'hdop': hdop,
            'easting': easting,
            'northing': northing,
            'zone_num': zone_num,
            'zone_letter': zone_letter
        }
    except (ValueError, IndexError):
        return None


def create_bag_from_txt(txt_file, bag_name, base_dir):
    """Create a ROS 2 bag from a .txt file"""
    print(f"\nCreating bag: {bag_name}")
    print("-" * 60)
    
    data_dir = base_dir / 'data'
    bag_path = data_dir / bag_name
    
    # Remove existing bag if present
    if bag_path.exists():
        import shutil
        shutil.rmtree(bag_path)
    
    # Create bag writer
    bag = ROS2BagWriter(bag_path)
    bag.create_topic('/rtk_data', 'std_msgs/msg/String')
    
    # Read and process data
    data_points = []
    with open(txt_file, 'r') as f:
        for line in f:
            parsed = parse_gngga(line)
            if parsed:
                data_points.append(parsed)
    
    if not data_points:
        print(f"  ERROR: No valid data found in {txt_file}")
        return
    
    print(f"  Parsed {len(data_points)} GNGGA messages")
    
    # Calculate timestamps (assuming 1 Hz rate)
    start_time_ns = int(time.time() * 1e9)
    
    for i, data in enumerate(data_points):
        timestamp_ns = start_time_ns + (i * int(1e9))  # 1 second intervals
        bag.write_message(data, timestamp_ns)
    
    end_time_ns = start_time_ns + (len(data_points) * int(1e9))
    
    # Update metadata and close
    bag.update_metadata(start_time_ns, end_time_ns)
    bag.close()
    
    # Print statistics
    avg_hdop = sum(d['hdop'] for d in data_points) / len(data_points)
    fix_types = {}
    for d in data_points:
        fix_types[d['fix_quality']] = fix_types.get(d['fix_quality'], 0) + 1
    
    print(f"  Created ROS 2 bag: {bag_path}")
    print(f"  Messages: {len(data_points)}")
    print(f"  Duration: {len(data_points)} seconds")
    print(f"  Average HDOP: {avg_hdop:.2f}")
    print(f"  Fix quality distribution: {fix_types}")
    
    # Show bag size
    db_size = (bag_path / 'lab2_master_0.db3').stat().st_size
    print(f"  Bag size: {db_size / 1024:.1f} KB")


def main():
    """Create ROS 2 bags for all RTK datasets"""
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / 'dataset'
    
    print("=" * 60)
    print("Creating ROS 2 Bag Files from RTK Data")
    print("=" * 60)
    
    datasets = [
        ('occludedRTK.txt', 'occluded_rtk'),
        ('openRTK.txt', 'open_rtk'),
        ('walkingRTK.txt', 'walking_rtk')
    ]
    
    for txt_name, bag_name in datasets:
        txt_file = dataset_dir / txt_name
        if txt_file.exists():
            create_bag_from_txt(txt_file, bag_name, base_dir)
        else:
            print(f"\nERROR: {txt_file} not found!")
    
    print("\n" + "=" * 60)
    print("ROS 2 Bag Creation Complete!")
    print("=" * 60)
    
    data_dir = base_dir / 'data'
    print(f"\nBag files location: {data_dir}")
    print("\nTo play a bag file in ROS 2:")
    print("  ros2 bag play data/occluded_rtk")
    print("  ros2 bag play data/open_rtk")
    print("  ros2 bag play data/walking_rtk")
    print("\nTo inspect bag info:")
    print("  ros2 bag info data/occluded_rtk")


if __name__ == '__main__':
    main()
