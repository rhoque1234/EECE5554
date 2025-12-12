#!/usr/bin/env python3
"""
Allan Variance Analysis for VectorNav IMU Gyroscope Data
Extracts noise parameters: Rate Random Walk (K), Angle Random Walk (N), and Bias Stability (B)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import allantools
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


class AllanVarianceAnalyzer:
    def __init__(self, bag_path):
        """Initialize Allan variance analyzer"""
        self.bag_path = Path(bag_path)
        self.gyro_data = []  # deg/s
        self.sample_rate = None
        
    def load_gyro_data(self):
        """Load gyroscope data from rosbag"""
        print(f"Loading gyro data from {self.bag_path}...")
        
        timestamps = []
        gyro_raw = []
        
        # Create type store
        typestore = get_typestore(Stores.LATEST)
        
        with Reader(self.bag_path) as reader:
            # Register message definitions
            for connection in reader.connections:
                if hasattr(connection, 'msgdef') and connection.msgdef:
                    from rosbags.typesys import get_types_from_msg
                    types = get_types_from_msg(connection.msgdef.data, connection.msgtype)
                    typestore.register(types)
            
            connections = [c for c in reader.connections if c.topic == '/imu']
            
            if not connections:
                print("No /imu topic found!")
                return False
            
            # Read messages
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                try:
                    # Deserialize message
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    
                    # Get timestamp
                    timestamps.append(timestamp / 1e9)  # Convert to seconds
                    
                    # Extract gyro data (convert rad/s to deg/s)
                    gyro_x = math.degrees(msg.imu.angular_velocity.x)
                    gyro_y = math.degrees(msg.imu.angular_velocity.y)
                    gyro_z = math.degrees(msg.imu.angular_velocity.z)
                    gyro_raw.append([gyro_x, gyro_y, gyro_z])
                    
                except Exception as e:
                    # Skip bad messages
                    print(f"Skipping bad message: {e}")
                    continue
        
        if len(timestamps) < 100:
            print("Not enough data!")
            return False
        
        # Calculate sample rate
        timestamps = np.array(timestamps)
        dt = np.diff(timestamps)
        self.sample_rate = 1.0 / np.median(dt)
        print(f"Sample rate: {self.sample_rate:.2f} Hz")
        
        self.gyro_data = np.array(gyro_raw)
        print(f"Loaded {len(self.gyro_data)} samples")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        
        return True
    
    def compute_allan_deviation(self, data, rate):
        """
        Compute Allan deviation using allantools
        
        Args:
            data: 1D array of gyro data (deg/s)
            rate: sample rate (Hz)
        
        Returns:
            tau: averaging times
            adev: Allan deviation values
        """
        # Compute Allan deviation
        tau, adev, adev_err, adev_n = allantools.oadev(data, rate=rate, 
                                                        data_type="freq", 
                                                        taus='octave')
        return tau, adev
    
    def extract_noise_parameters(self, tau, adev):
        """
        Extract noise parameters from Allan deviation plot
        
        Args:
            tau: averaging times
            adev: Allan deviation values
        
        Returns:
            dict with N (angle random walk), B (bias stability), K (rate random walk)
        """
        params = {}
        
        # Angle Random Walk (N): slope = -1/2, read at tau = 1 second
        # Find the value at tau = 1 or interpolate
        if 1.0 in tau:
            idx = np.where(tau == 1.0)[0][0]
            params['N'] = adev[idx]
        else:
            # Interpolate in log-log space
            log_tau = np.log10(tau)
            log_adev = np.log10(adev)
            params['N'] = 10 ** np.interp(0, log_tau, log_adev)  # tau = 10^0 = 1
        
        # Bias Stability (B): minimum of the Allan deviation curve
        min_idx = np.argmin(adev)
        params['B'] = adev[min_idx]
        params['B_tau'] = tau[min_idx]
        
        # Rate Random Walk (K): slope = +1/2, read at tau = 3 seconds (or extrapolate)
        # K = adev(tau) * sqrt(3) at the +1/2 slope region
        # Typically appears at larger tau values
        if len(tau) > 3:
            # Use the last few points which typically show +1/2 slope
            late_idx = -3
            params['K'] = adev[late_idx] * np.sqrt(3.0 / tau[late_idx])
        else:
            params['K'] = np.nan
        
        return params
    
    def plot_allan_deviation(self, save_path='fig3_allan_variance.png'):
        """
        Figure 3: Plot Allan deviation for gyro x, y, z
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle('Gyroscope Allan Deviation', fontsize=14, fontweight='bold')
        
        labels = ['X-axis', 'Y-axis', 'Z-axis']
        colors = ['r', 'g', 'b']
        
        all_params = []
        
        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            print(f"\nComputing Allan deviation for Gyro {label}...")
            
            # Compute Allan deviation
            tau, adev = self.compute_allan_deviation(self.gyro_data[:, i], self.sample_rate)
            
            # Extract parameters
            params = self.extract_noise_parameters(tau, adev)
            all_params.append(params)
            
            print(f"  Angle Random Walk (N): {params['N']:.6f} °/√hr")
            print(f"  Bias Stability (B): {params['B']:.6f} °/hr at tau={params['B_tau']:.1f}s")
            if not np.isnan(params['K']):
                print(f"  Rate Random Walk (K): {params['K']:.6f} °/√hr")
            
            # Plot
            ax.loglog(tau, adev, color=color, linewidth=2, label='Allan Deviation')
            
            # Mark key points
            ax.plot(1.0, params['N'], 'ko', markersize=8, label=f"N={params['N']:.4f} °/√hr")
            ax.plot(params['B_tau'], params['B'], 'ks', markersize=8, 
                   label=f"B={params['B']:.4f} °/hr")
            
            # Add reference slopes
            tau_ref = np.array([tau[0], tau[-1]])
            
            # -1/2 slope (Angle Random Walk)
            arw_line = params['N'] * (tau_ref / 1.0) ** (-0.5)
            ax.loglog(tau_ref, arw_line, 'k--', alpha=0.5, linewidth=1, label='Slope -1/2')
            
            # +1/2 slope (Rate Random Walk)
            if not np.isnan(params['K']):
                rrw_line = params['K'] * (tau_ref / 1.0) ** (0.5) / np.sqrt(3)
                ax.loglog(tau_ref, rrw_line, 'k:', alpha=0.5, linewidth=1, label='Slope +1/2')
            
            ax.set_xlabel('Averaging Time τ (s)', fontsize=10)
            ax.set_ylabel('Allan Deviation (°/s)', fontsize=10)
            ax.set_title(f'Gyro {label}')
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved {save_path}")
        plt.close()
        
        return all_params
    
    def save_parameters(self, params, save_path='allan_parameters.txt'):
        """Save extracted noise parameters to text file"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("VectorNav VN-100 Gyroscope Noise Parameters\n")
            f.write("=" * 60 + "\n\n")
            
            for i, (axis, p) in enumerate(zip(['X', 'Y', 'Z'], params)):
                f.write(f"Gyro {axis}-axis:\n")
                f.write(f"  Angle Random Walk (N): {p['N']:.6f} °/√hr\n")
                f.write(f"  Bias Stability (B): {p['B']:.6f} °/hr (at τ={p['B_tau']:.1f}s)\n")
                if not np.isnan(p['K']):
                    f.write(f"  Rate Random Walk (K): {p['K']:.6f} °/√hr\n")
                f.write("\n")
        
        print(f"Parameters saved to {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Allan Variance Analysis for IMU Gyro Data')
    parser.add_argument('bag_path', help='Path to rosbag directory (5 hour stationary data)')
    parser.add_argument('--output', '-o', default='analysis',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = AllanVarianceAnalyzer(args.bag_path)
    
    # Load data
    if not analyzer.load_gyro_data():
        print("Failed to load data")
        return
    
    # Generate Allan deviation plot
    print("\nComputing Allan variance...")
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    params = analyzer.plot_allan_deviation(output_dir / 'fig3_allan_variance.png')
    analyzer.save_parameters(params, output_dir / 'allan_parameters.txt')
    
    print("\nAllan variance analysis complete!")


if __name__ == '__main__':
    main()
