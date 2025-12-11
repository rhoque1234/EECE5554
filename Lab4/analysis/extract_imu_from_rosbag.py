#!/usr/bin/env python3
"""
Extract IMU data from rosbag2 files for analysis
"""

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
import argparse


def extract_imu_data(rosbag_path):
    """Extract IMU and magnetometer data from rosbag"""
    
    rosbag_path = Path(rosbag_path)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    # Lists to store data
    times = []
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    mag_x, mag_y, mag_z = [], [], []
    orient_x, orient_y, orient_z, orient_w = [], [], [], []
    
    print(f"Reading rosbag from {rosbag_path}...")
    
    with Reader(rosbag_path) as reader:
        # Read IMU messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/imu/data':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                # Convert timestamp to seconds
                time_sec = timestamp / 1e9
                times.append(time_sec)
                
                # Extract accelerometer data
                accel_x.append(msg.linear_acceleration.x)
                accel_y.append(msg.linear_acceleration.y)
                accel_z.append(msg.linear_acceleration.z)
                
                # Extract gyroscope data
                gyro_x.append(msg.angular_velocity.x)
                gyro_y.append(msg.angular_velocity.y)
                gyro_z.append(msg.angular_velocity.z)
                
                # Extract orientation
                orient_x.append(msg.orientation.x)
                orient_y.append(msg.orientation.y)
                orient_z.append(msg.orientation.z)
                orient_w.append(msg.orientation.w)
                
            elif connection.topic == '/imu/mag':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                # Extract magnetometer data
                mag_x.append(msg.magnetic_field.x)
                mag_y.append(msg.magnetic_field.y)
                mag_z.append(msg.magnetic_field.z)
    
    # Convert to numpy arrays
    data = {
        'time': np.array(times) - times[0],  # Start from 0
        'accel_x': np.array(accel_x),
        'accel_y': np.array(accel_y),
        'accel_z': np.array(accel_z),
        'gyro_x': np.array(gyro_x),
        'gyro_y': np.array(gyro_y),
        'gyro_z': np.array(gyro_z),
        'mag_x': np.array(mag_x),
        'mag_y': np.array(mag_y),
        'mag_z': np.array(mag_z),
        'orientation_x': np.array(orient_x),
        'orientation_y': np.array(orient_y),
        'orientation_z': np.array(orient_z),
        'orientation_w': np.array(orient_w),
    }
    
    print(f"Extracted {len(times)} IMU samples")
    print(f"Duration: {data['time'][-1]:.1f} seconds")
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Extract IMU data from rosbag')
    parser.add_argument('--input', required=True, help='Input rosbag directory')
    parser.add_argument('--output', required=True, help='Output NPZ file path')
    
    args = parser.parse_args()
    
    data = extract_imu_data(args.input)
    
    # Save to NPZ file
    output_path = Path(args.output)
    np.savez_compressed(
        output_path,
        time=data['time'],
        accel_x=data['accel_x'],
        accel_y=data['accel_y'],
        accel_z=data['accel_z'],
        gyro_x=data['gyro_x'],
        gyro_y=data['gyro_y'],
        gyro_z=data['gyro_z'],
        mag_x=data['mag_x'],
        mag_y=data['mag_y'],
        mag_z=data['mag_z'],
        orientation_x=data['orientation_x'],
        orientation_y=data['orientation_y'],
        orientation_z=data['orientation_z'],
        orientation_w=data['orientation_w']
    )
    
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
