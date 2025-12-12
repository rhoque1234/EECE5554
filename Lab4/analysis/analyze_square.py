#!/usr/bin/env python3
"""
Analysis script for Lab 4 - Square Walking Data
Generates Figures 9-16 as required by lab instructions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import cumulative_trapezoid
import argparse


class SquareWalkingAnalyzer:
    def __init__(self, data_path):
        """Initialize analyzer with data path"""
        self.data_path = Path(data_path)
        self.load_data()
        
    def load_data(self):
        """Load numpy data"""
        print(f"Loading data from {self.data_path}...")
        data = np.load(self.data_path / 'imu_data.npz')
        
        self.time = data['time']
        self.accel_x = data['accel_x']
        self.accel_y = data['accel_y']
        self.accel_z = data['accel_z']
        self.gyro_x = data['gyro_x']
        self.gyro_y = data['gyro_y']
        self.gyro_z = data['gyro_z']
        self.mag_x = data['mag_x']
        self.mag_y = data['mag_y']
        self.mag_z = data['mag_z']
        
        print(f"Loaded {len(self.time)} samples ({self.time[-1]:.1f} seconds)")
        
    def calibrate_magnetometer(self):
        """
        Apply hard iron calibration to magnetometer data
        Returns calibrated mag_x and mag_y, and calibration parameters
        """
        # Find hard iron offset (center of circular pattern)
        offset_x = (np.max(self.mag_x) + np.min(self.mag_x)) / 2
        offset_y = (np.max(self.mag_y) + np.min(self.mag_y)) / 2
        
        # Apply calibration
        mag_x_cal = self.mag_x - offset_x
        mag_y_cal = self.mag_y - offset_y
        
        return mag_x_cal, mag_y_cal, offset_x, offset_y
        
    def plot_mag_calibration(self, output_dir):
        """
        Figure 1: Magnetometer X vs Y before and after calibration
        """
        mag_x_cal, mag_y_cal, offset_x, offset_y = self.calibrate_magnetometer()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Before calibration
        ax1.scatter(self.mag_x * 1e6, self.mag_y * 1e6, c=self.time, cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel('Magnetic Field X (µT)')
        ax1.set_ylabel('Magnetic Field Y (µT)')
        ax1.set_title('Before Calibration')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # After calibration
        scatter = ax2.scatter(mag_x_cal * 1e6, mag_y_cal * 1e6, c=self.time, cmap='viridis', s=1, alpha=0.6)
        ax2.set_xlabel('Magnetic Field X (µT)')
        ax2.set_ylabel('Magnetic Field Y (µT)')
        ax2.set_title('After Calibration')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.colorbar(scatter, ax=ax2, label='Time (s)')
        plt.suptitle('Figure 9: Magnetometer Calibration (Square Walking)', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fig9_mag_calibration.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
        return mag_x_cal, mag_y_cal
        
    def plot_gyro_x(self, output_dir):
        """
        Figure 2: Gyroscope X - rate and integrated rotation
        """
        # Integrate gyro to get rotation
        rotation_x = cumulative_trapezoid(self.gyro_x, self.time, initial=0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Rotational rate
        ax1.plot(self.time, self.gyro_x, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.set_title('Rotational Rate around X-axis')
        ax1.grid(True, alpha=0.3)
        
        # Integrated rotation
        ax2.plot(self.time, np.degrees(rotation_x), 'r-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Rotation (degrees)')
        ax2.set_title('Integrated Rotation around X-axis')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Figure 10: Gyroscope X-axis Analysis', fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fig10_gyro_x.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
    def plot_gyro_y(self, output_dir):
        """
        Figure 3: Gyroscope Y - rate and integrated rotation
        """
        rotation_y = cumulative_trapezoid(self.gyro_y, self.time, initial=0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(self.time, self.gyro_y, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.set_title('Rotational Rate around Y-axis')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.time, np.degrees(rotation_y), 'r-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Rotation (degrees)')
        ax2.set_title('Integrated Rotation around Y-axis')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Figure 11: Gyroscope Y-axis Analysis', fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fig11_gyro_y.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
    def plot_gyro_z_and_heading(self, output_dir):
        """
        Figure 4: Gyroscope Z - rate, integrated rotation, and magnetometer heading
        """
        # Integrate gyro Z to get heading
        rotation_z = cumulative_trapezoid(self.gyro_z, self.time, initial=0)
        
        # Calibrate magnetometer
        mag_x_cal, mag_y_cal, _, _ = self.calibrate_magnetometer()
        
        # Calculate magnetometer heading
        mag_heading = np.arctan2(mag_y_cal, mag_x_cal)
        mag_heading_unwrapped = np.unwrap(mag_heading)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Rotational rate
        ax1.plot(self.time, self.gyro_z, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.set_title('Rotational Rate around Z-axis (Yaw Rate)')
        ax1.grid(True, alpha=0.3)
        
        # Integrated rotation from gyro
        ax2.plot(self.time, np.degrees(rotation_z), 'r-', linewidth=0.5, label='Gyro Integration')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Rotation (degrees)')
        ax2.set_title('Integrated Rotation from Gyroscope')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Magnetometer heading
        ax3.plot(self.time, np.degrees(mag_heading_unwrapped), 'g-', linewidth=0.5, label='Magnetometer')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Heading (degrees)')
        ax3.set_title('Heading from Magnetometer (atan2(Y/X))')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.suptitle('Figure 12: Z-axis Rotation and Heading Analysis', fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fig12_gyro_z_heading.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
        return rotation_z, mag_heading_unwrapped
        
    def plot_accel_x(self, output_dir):
        """
        Figure 13: Accelerometer X - acceleration, velocity, and displacement
        """
        # Integrate to get velocity
        velocity_x = cumulative_trapezoid(self.accel_x, self.time, initial=0)
        # Integrate velocity to get displacement
        displacement_x = cumulative_trapezoid(velocity_x, self.time, initial=0)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        ax1.plot(self.time, self.accel_x, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title('Linear Acceleration in X-axis (Forward)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.time, velocity_x, 'r-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Integrated Velocity in X-axis')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(self.time, displacement_x, 'g-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Displacement (m)')
        ax3.set_title('Integrated Displacement in X-axis')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Figure 13: Accelerometer X-axis Analysis', fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fig13_accel_x.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
        return velocity_x
        
    def plot_accel_y(self, output_dir):
        """
        Figure 14: Accelerometer Y - acceleration, velocity, and displacement
        """
        velocity_y = cumulative_trapezoid(self.accel_y, self.time, initial=0)
        # Integrate velocity to get displacement
        displacement_y = cumulative_trapezoid(velocity_y, self.time, initial=0)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        ax1.plot(self.time, self.accel_y, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title('Linear Acceleration in Y-axis (Lateral)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.time, velocity_y, 'r-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Integrated Velocity in Y-axis')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(self.time, displacement_y, 'g-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Displacement (m)')
        ax3.set_title('Integrated Displacement in Y-axis')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Figure 14: Accelerometer Y-axis Analysis', fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fig14_accel_y.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
        return velocity_y
        
    def plot_accel_z(self, output_dir):
        """
        Figure 15: Accelerometer Z - acceleration, velocity, and displacement
        """
        # Remove gravity (9.81 m/s²) before integrating
        accel_z_no_gravity = self.accel_z - 9.81
        velocity_z = cumulative_trapezoid(accel_z_no_gravity, self.time, initial=0)
        # Integrate velocity to get displacement
        displacement_z = cumulative_trapezoid(velocity_z, self.time, initial=0)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        ax1.plot(self.time, self.accel_z, 'b-', linewidth=0.5, label='Raw')
        ax1.plot(self.time, accel_z_no_gravity, 'r-', linewidth=0.5, alpha=0.7, label='Gravity Removed')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title('Linear Acceleration in Z-axis (Vertical)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(self.time, velocity_z, 'r-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Integrated Velocity in Z-axis')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(self.time, displacement_z, 'g-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Displacement (m)')
        ax3.set_title('Integrated Displacement in Z-axis')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Figure 15: Accelerometer Z-axis Analysis', fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fig15_accel_z.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
        return velocity_z
        
    def plot_trajectory(self, output_dir):
        """
        Figure 8: North vs East position using magnetometer and gyro heading
        """
        # Compute rotation from gyro
        rotation_z = cumulative_trapezoid(self.gyro_z, self.time, initial=0)
        
        # Compute magnetometer heading
        mag_x_cal, mag_y_cal, _, _ = self.calibrate_magnetometer()
        mag_heading = np.arctan2(mag_y_cal, mag_x_cal)
        mag_heading_unwrapped = np.unwrap(mag_heading)
        
        # Compute velocity in X (body frame)
        velocity_x = cumulative_trapezoid(self.accel_x, self.time, initial=0)
        
        # Convert body frame velocity to world frame and integrate
        # Using magnetometer heading
        pos_n_mag = np.zeros(len(self.time))
        pos_e_mag = np.zeros(len(self.time))
        
        for i in range(1, len(self.time)):
            dt = self.time[i] - self.time[i-1]
            heading_mag = mag_heading_unwrapped[i]
            
            # Transform velocity from body to world frame
            vn_mag = velocity_x[i] * np.cos(heading_mag)
            ve_mag = velocity_x[i] * np.sin(heading_mag)
            
            pos_n_mag[i] = pos_n_mag[i-1] + vn_mag * dt
            pos_e_mag[i] = pos_e_mag[i-1] + ve_mag * dt
            
        # Using gyro heading
        pos_n_gyro = np.zeros(len(self.time))
        pos_e_gyro = np.zeros(len(self.time))
        
        for i in range(1, len(self.time)):
            dt = self.time[i] - self.time[i-1]
            heading_gyro = rotation_z[i]
            
            vn_gyro = velocity_x[i] * np.cos(heading_gyro)
            ve_gyro = velocity_x[i] * np.sin(heading_gyro)
            
            pos_n_gyro[i] = pos_n_gyro[i-1] + vn_gyro * dt
            pos_e_gyro[i] = pos_e_gyro[i-1] + ve_gyro * dt
            
        # Plot trajectories
        plt.figure(figsize=(10, 10))
        plt.plot(pos_e_mag, pos_n_mag, 'b-', linewidth=1, alpha=0.7, label='Magnetometer Heading')
        plt.plot(pos_e_gyro, pos_n_gyro, 'r--', linewidth=1, alpha=0.7, label='Gyro Heading')
        plt.scatter([0], [0], c='green', s=100, marker='o', label='Start', zorder=5)
        plt.xlabel('East Position (m)')
        plt.ylabel('North Position (m)')
        plt.title('Figure 16: Estimated Trajectory (Square Walking)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fig16_trajectory.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
    def generate_all_plots(self, output_dir):
        """Generate all required plots for square walking"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Generating plots for square walking...")
        print("=" * 60)
        
        self.plot_mag_calibration(output_dir)
        self.plot_gyro_x(output_dir)
        self.plot_gyro_y(output_dir)
        gyro_z, mag_heading = self.plot_gyro_z_and_heading(output_dir)
        velocity_x = self.plot_accel_x(output_dir)
        velocity_y = self.plot_accel_y(output_dir)
        velocity_z = self.plot_accel_z(output_dir)
        self.plot_trajectory(output_dir)
        
        print("=" * 60)
        print(f"All plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze square walking data for Lab 4')
    parser.add_argument('--data', required=True, help='Path to data directory')
    parser.add_argument('--output', required=True, help='Output directory for plots')
    
    args = parser.parse_args()
    
    analyzer = SquareWalkingAnalyzer(args.data)
    analyzer.generate_all_plots(args.output)


if __name__ == '__main__':
    main()
