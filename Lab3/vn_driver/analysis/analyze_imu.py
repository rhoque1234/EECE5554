#!/usr/bin/env python3
"""
Analysis script for VectorNav IMU data
Generates required plots for Lab 3
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


class IMUAnalyzer:
    def __init__(self, bag_path):
        """Initialize analyzer with rosbag path"""
        self.bag_path = Path(bag_path)
        self.timestamps = []
        self.gyro_data = []  # rad/s
        self.accel_data = []  # m/s^2
        self.orientation_data = []  # degrees (roll, pitch, yaw)
        self.mag_data = []  # Tesla
        
    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    
    def load_rosbag(self):
        """Load IMU data from rosbag"""
        print(f"Loading rosbag from {self.bag_path}...")
        
        # Create type store and register types from bag
        typestore = get_typestore(Stores.LATEST)
        
        with Reader(self.bag_path) as reader:
            # Register message definitions
            for connection in reader.connections:
                if hasattr(connection, 'msgdef') and connection.msgdef:
                    from rosbags.typesys import get_types_from_msg
                    types = get_types_from_msg(connection.msgdef.data, connection.msgtype)
                    typestore.register(types)
            
            # Get topic info
            connections = [c for c in reader.connections if c.topic == '/imu']
            
            if not connections:
                print("No /imu topic found in bag!")
                return False
            
            # Read messages
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # Deserialize message
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                # Store timestamp (convert to seconds from start)
                if len(self.timestamps) == 0:
                    self.start_time = timestamp
                t = (timestamp - self.start_time) / 1e9  # nanoseconds to seconds
                self.timestamps.append(t)
                
                # Extract gyro data (convert rad/s to deg/s for analysis)
                gyro_x = math.degrees(msg.imu.angular_velocity.x)
                gyro_y = math.degrees(msg.imu.angular_velocity.y)
                gyro_z = math.degrees(msg.imu.angular_velocity.z)
                self.gyro_data.append([gyro_x, gyro_y, gyro_z])
                
                # Extract acceleration data (m/s^2)
                accel_x = msg.imu.linear_acceleration.x
                accel_y = msg.imu.linear_acceleration.y
                accel_z = msg.imu.linear_acceleration.z
                self.accel_data.append([accel_x, accel_y, accel_z])
                
                # Convert quaternion to Euler angles
                quat = msg.imu.orientation
                roll, pitch, yaw = self.quaternion_to_euler(quat.x, quat.y, quat.z, quat.w)
                self.orientation_data.append([roll, pitch, yaw])
                
                # Extract magnetometer data (Tesla)
                mag_x = msg.mag_field.magnetic_field.x
                mag_y = msg.mag_field.magnetic_field.y
                mag_z = msg.mag_field.magnetic_field.z
                self.mag_data.append([mag_x, mag_y, mag_z])
        
        # Convert to numpy arrays
        self.timestamps = np.array(self.timestamps)
        self.gyro_data = np.array(self.gyro_data)
        self.accel_data = np.array(self.accel_data)
        self.orientation_data = np.array(self.orientation_data)
        self.mag_data = np.array(self.mag_data)
        
        print(f"Loaded {len(self.timestamps)} samples")
        print(f"Duration: {self.timestamps[-1]:.2f} seconds")
        
        return True
    
    def plot_gyro(self, save_path='fig0_gyro.png'):
        """Figure 0: Plot gyroscope data"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Gyroscope Data (Rotational Rate)', fontsize=14, fontweight='bold')
        
        labels = ['X-axis', 'Y-axis', 'Z-axis']
        colors = ['r', 'g', 'b']
        
        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            ax.plot(self.timestamps, self.gyro_data[:, i], color=color, linewidth=0.5)
            ax.set_ylabel(f'Angular Rate (°/s)', fontsize=10)
            ax.set_title(f'Gyro {label}')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
        plt.close()
    
    def plot_accel(self, save_path='fig1_accel.png'):
        """Figure 1: Plot accelerometer data"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Accelerometer Data', fontsize=14, fontweight='bold')
        
        labels = ['X-axis', 'Y-axis', 'Z-axis']
        colors = ['r', 'g', 'b']
        
        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            ax.plot(self.timestamps, self.accel_data[:, i], color=color, linewidth=0.5)
            ax.set_ylabel(f'Acceleration (m/s²)', fontsize=10)
            ax.set_title(f'Accel {label}')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
        plt.close()
    
    def plot_orientation(self, save_path='fig2_orientation.png'):
        """Figure 2: Plot orientation from VN estimation"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Orientation (from VectorNav Estimation)', fontsize=14, fontweight='bold')
        
        labels = ['Roll', 'Pitch', 'Yaw']
        colors = ['r', 'g', 'b']
        
        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            ax.plot(self.timestamps, self.orientation_data[:, i], color=color, linewidth=0.5)
            ax.set_ylabel(f'{label} (°)', fontsize=10)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
        plt.close()
    
    def plot_motion_clip(self, start_time, end_time, save_path, title='Motion Clip'):
        """Plot a short clip of interesting motion"""
        # Find indices for time range
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        t = self.timestamps[mask] - start_time  # Reset time to 0
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Plot gyro
        axes[0].plot(t, self.gyro_data[mask, 0], 'r-', label='X', linewidth=1)
        axes[0].plot(t, self.gyro_data[mask, 1], 'g-', label='Y', linewidth=1)
        axes[0].plot(t, self.gyro_data[mask, 2], 'b-', label='Z', linewidth=1)
        axes[0].set_ylabel('Gyro (°/s)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Gyroscope')
        
        # Plot accel
        axes[1].plot(t, self.accel_data[mask, 0], 'r-', label='X', linewidth=1)
        axes[1].plot(t, self.accel_data[mask, 1], 'g-', label='Y', linewidth=1)
        axes[1].plot(t, self.accel_data[mask, 2], 'b-', label='Z', linewidth=1)
        axes[1].set_ylabel('Accel (m/s²)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Accelerometer')
        
        # Plot orientation
        axes[2].plot(t, self.orientation_data[mask, 0], 'r-', label='Roll', linewidth=1)
        axes[2].plot(t, self.orientation_data[mask, 1], 'g-', label='Pitch', linewidth=1)
        axes[2].plot(t, self.orientation_data[mask, 2], 'b-', label='Yaw', linewidth=1)
        axes[2].set_ylabel('Orientation (°)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Orientation')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
        plt.close()
    
    def generate_all_plots(self, output_dir='.'):
        """Generate all required plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Figures 0-2: Full data plots
        self.plot_gyro(output_dir / 'fig0_gyro.png')
        self.plot_accel(output_dir / 'fig1_accel.png')
        self.plot_orientation(output_dir / 'fig2_orientation.png')
        
        # Figures 4-6: Motion clips (assuming motion data exists)
        duration = self.timestamps[-1]
        
        # Find interesting sections (look for high variance)
        if duration > 15:  # If we have enough data
            # Clip 1: First interesting section
            self.plot_motion_clip(10, 15, output_dir / 'fig4_motion_clip1.png', 
                                'Motion Clip 1 (10-15s)')
            
            # Clip 2: Middle section
            mid_time = duration / 2
            self.plot_motion_clip(mid_time, mid_time + 5, output_dir / 'fig5_motion_clip2.png',
                                f'Motion Clip 2 ({mid_time:.0f}-{mid_time+5:.0f}s)')
            
            # Clip 3: Near end
            self.plot_motion_clip(duration - 15, duration - 10, output_dir / 'fig6_motion_clip3.png',
                                f'Motion Clip 3 ({duration-15:.0f}-{duration-10:.0f}s)')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze VectorNav IMU data')
    parser.add_argument('bag_path', help='Path to rosbag directory')
    parser.add_argument('--output', '-o', default='analysis', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = IMUAnalyzer(args.bag_path)
    
    # Load data
    if not analyzer.load_rosbag():
        print("Failed to load rosbag")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    analyzer.generate_all_plots(args.output)
    
    print("\nAnalysis complete!")
    print(f"Plots saved to {args.output}/")


if __name__ == '__main__':
    main()
