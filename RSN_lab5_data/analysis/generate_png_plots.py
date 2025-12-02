#!/usr/bin/env python3
"""
Generate PNG versions of all plots using matplotlib
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def parse_vnymr_string(vnymr_str):
    """Parse VNYMR NMEA string to extract sensor data"""
    try:
        parts = vnymr_str.split(',')
        if len(parts) >= 13 and parts[0] == '$VNYMR':
            return {
                'yaw': float(parts[1]),
                'pitch': float(parts[2]),
                'roll': float(parts[3]),
                'mag_x': float(parts[4]),
                'mag_y': float(parts[5]),
                'mag_z': float(parts[6]),
                'accel_x': float(parts[7]),
                'accel_y': float(parts[8]),
                'accel_z': float(parts[9]),
                'gyro_x': float(parts[10]),
                'gyro_y': float(parts[11]),
                'gyro_z': float(parts[12].split('*')[0])
            }
    except (ValueError, IndexError):
        pass
    return None

def load_imu_data(filepath):
    """Load IMU data from CSV"""
    df = pd.read_csv(filepath, header=None)
    rows_data = []
    for idx, row in df.iterrows():
        vnymr_parts = [str(row[i]) for i in range(58, 71) if pd.notna(row[i])]
        if vnymr_parts:
            vnymr_str = ','.join(vnymr_parts)
            parsed = parse_vnymr_string(vnymr_str)
            if parsed:
                parsed['time'] = float(row[1])
                rows_data.append(parsed)
    result = pd.DataFrame(rows_data)
    result['time'] = result['time'] - result['time'].iloc[0]
    result.set_index('time', inplace=True)
    return result

def load_gps_data(filepath):
    """Load GPS data from CSV"""
    df = pd.read_csv(filepath, header=None, usecols=[0,1,3,4,5,6,7])
    df.columns = ['seq', 'time', 'lat', 'lon', 'alt', 'utm_e', 'utm_n']
    df['time'] = df['time'] - df['time'].iloc[0]
    df.set_index('time', inplace=True)
    return df

def calibrate_magnetometer(mag_data):
    """Calculate hard-iron and soft-iron calibration"""
    bias = mag_data.mean()
    mag_centered = mag_data - bias
    scale_factors = 1.0 / mag_centered.std()
    return bias, scale_factors

def apply_calibration(mag_data, bias, scale_factors):
    """Apply calibration to magnetometer data"""
    return (mag_data - bias) * scale_factors

def butter_filter(data, cutoff, fs, order=5, btype='low'):
    """Apply Butterworth filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype)
    return filtfilt(b, a, data)

print("Loading data...")
imu_circles = load_imu_data('../data/imugoing_in_circles1.csv')
imu_driving = load_imu_data('../data/imudriving1.csv')
gps_circles = load_gps_data('../data/gpsgoing_in_circles1.csv')
gps_driving = load_gps_data('../data/gpsdriving1.csv')

print("Calibrating magnetometer...")
mag_circles = imu_circles[['mag_x', 'mag_y', 'mag_z']]
bias, scale_factors = calibrate_magnetometer(mag_circles)
mag_calibrated = apply_calibration(mag_circles, bias, scale_factors)

# Plot 0: Magnetometer Calibration
print("Generating Plot 0...")
plt.figure(figsize=(10, 6))
plt.scatter(mag_circles['mag_x'], mag_circles['mag_y'], alpha=0.5, s=10, label='Uncalibrated')
plt.scatter(mag_calibrated['mag_x'], mag_calibrated['mag_y'], alpha=0.5, s=10, label='Calibrated')
plt.xlabel('Mag X (µT)')
plt.ylabel('Mag Y (µT)')
plt.title('Plot 0: Magnetometer Calibration (XY Plane)')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../docs/plot0_mag_calibration.png', dpi=150, bbox_inches='tight')
plt.close()

# Apply calibration to driving data
mag_driving = imu_driving[['mag_x', 'mag_y', 'mag_z']]
mag_driving_cal = apply_calibration(mag_driving, bias, scale_factors)

# Compute yaw angles
yaw_uncal = np.arctan2(-mag_driving['mag_y'], mag_driving['mag_x'])
yaw_cal = np.arctan2(-mag_driving_cal['mag_y'], mag_driving_cal['mag_x'])

# Plot 1: Mag Yaw Comparison
print("Generating Plot 1...")
plt.figure(figsize=(12, 5))
plt.plot(imu_driving.index, np.rad2deg(yaw_uncal), label='Uncalibrated Yaw', alpha=0.7)
plt.plot(imu_driving.index, np.rad2deg(yaw_cal), label='Calibrated Yaw', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Yaw (degrees)')
plt.title('Plot 1: Magnetometer Yaw Estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../docs/plot1_mag_yaw.png', dpi=150, bbox_inches='tight')
plt.close()

# Gyro integration
dt = imu_driving.index.to_series().diff().fillna(0)
gyro_yaw_rad = np.cumsum(imu_driving['gyro_z'] * dt)

# Plot 2: Gyro Yaw
print("Generating Plot 2...")
plt.figure(figsize=(12, 5))
plt.plot(imu_driving.index, np.rad2deg(gyro_yaw_rad), label='Gyro Integrated Yaw')
plt.xlabel('Time (s)')
plt.ylabel('Yaw (degrees)')
plt.title('Plot 2: Gyro Yaw Estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../docs/plot2_gyro_yaw.png', dpi=150, bbox_inches='tight')
plt.close()

# Complementary filter
cutoff_freq = 0.1
fs = 1.0 / dt.mean()
# Ensure cutoff is valid (less than Nyquist frequency)
nyq = 0.5 * fs
if cutoff_freq >= nyq:
    cutoff_freq = nyq * 0.9  # Use 90% of Nyquist if too high
mag_lp = butter_filter(yaw_cal, cutoff_freq, fs, btype='low')
gyro_hp = butter_filter(gyro_yaw_rad, cutoff_freq, fs, btype='high')
comp_yaw = mag_lp + gyro_hp

# Plot 3: Complementary Filter Analysis
print("Generating Plot 3...")
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(imu_driving.index, np.rad2deg(yaw_cal), label='Magnetometer')
ax1.set_ylabel('Yaw (degrees)')
ax1.set_title('Magnetometer Yaw')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(imu_driving.index, np.rad2deg(gyro_yaw_rad), label='Gyroscope', color='orange')
ax2.set_ylabel('Yaw (degrees)')
ax2.set_title('Gyroscope Yaw')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(imu_driving.index, np.rad2deg(mag_lp), label='Low-pass (Mag)', color='green')
ax3.plot(imu_driving.index, np.rad2deg(gyro_hp), label='High-pass (Gyro)', color='red', alpha=0.7)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Yaw (degrees)')
ax3.set_title('Filter Components')
ax3.grid(True, alpha=0.3)
ax3.legend()

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(imu_driving.index, np.rad2deg(comp_yaw), label='Fused', color='purple')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Yaw (degrees)')
ax4.set_title('Complementary Filter Output')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.suptitle('Plot 3: Complementary Filter Analysis', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('../docs/plot3_complementary_filter.png', dpi=150, bbox_inches='tight')
plt.close()

# Velocity from accelerometer
accel_forward = imu_driving['accel_x'] * np.cos(comp_yaw) - imu_driving['accel_y'] * np.sin(comp_yaw)
vel_accel = cumulative_trapezoid(accel_forward - accel_forward.mean(), imu_driving.index, initial=0)

# Plot 4: Accel Velocity
print("Generating Plot 4...")
plt.figure(figsize=(12, 5))
plt.plot(imu_driving.index, vel_accel, label='Velocity from Accel')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Plot 4: Forward Velocity from Accelerometer')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../docs/plot4_accel_velocity.png', dpi=150, bbox_inches='tight')
plt.close()

# Velocity from GPS
dx = gps_driving['utm_e'].diff()
dy = gps_driving['utm_n'].diff()
dt_gps = gps_driving.index.to_series().diff()
vel_gps = np.sqrt(dx**2 + dy**2) / dt_gps

# Plot 5: GPS Velocity
print("Generating Plot 5...")
plt.figure(figsize=(12, 5))
plt.plot(gps_driving.index[1:], vel_gps[1:], label='Velocity from GPS')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Plot 5: Forward Velocity from GPS')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../docs/plot5_gps_velocity.png', dpi=150, bbox_inches='tight')
plt.close()

# Trajectory estimation
vx_imu = vel_accel * np.cos(comp_yaw)
vy_imu = vel_accel * np.sin(comp_yaw)
x_imu = cumulative_trapezoid(vx_imu, imu_driving.index, initial=0)
y_imu = cumulative_trapezoid(vy_imu, imu_driving.index, initial=0)

# GPS trajectory (relative)
x_gps = gps_driving['utm_e'] - gps_driving['utm_e'].iloc[0]
y_gps = gps_driving['utm_n'] - gps_driving['utm_n'].iloc[0]

# Plot 6: Trajectory Comparison
print("Generating Plot 6...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(x_imu, y_imu, label='IMU Trajectory', alpha=0.7)
ax1.set_xlabel('East (m)')
ax1.set_ylabel('North (m)')
ax1.set_title('IMU-based Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

ax2.plot(x_gps, y_gps, label='GPS Trajectory', color='orange', alpha=0.7)
ax2.set_xlabel('East (m)')
ax2.set_ylabel('North (m)')
ax2.set_title('GPS-based Trajectory')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.suptitle('Plot 6: Trajectory Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('../docs/plot6_trajectory_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("All PNG plots generated successfully!")
print("Saved to ../docs/ directory")
print("="*60)
