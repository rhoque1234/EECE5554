#!/usr/bin/env python3
"""
EECE 5554 Lab 5: NUance Navigation Analysis
Complete analysis script for IMU and GPS data
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

def parse_vnymr_string(vnymr_str):
    """Parse VNYMR NMEA string to extract sensor data"""
    # Format: $VNYMR,yaw,pitch,roll,magX,magY,magZ,accelX,accelY,accelZ,gyroX,gyroY,gyroZ*checksum
    try:
        parts = vnymr_str.split(',')
        if len(parts) >= 13 and parts[0] == '$VNYMR':
            return {
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
    except:
        pass
    return None

def load_imu_data(filepath):
    """Load IMU data from CSV file"""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, header=None, usecols=list(range(0, 2)) + list(range(58, 71)))
    
    data_list = []
    for _, row in df.iterrows():
        # Reconstruct VNYMR string from columns 58-70
        vnymr_str = ','.join([str(row.iloc[i]) for i in range(2, len(row))])
        if vnymr_str.startswith('$VNYMR'):
            parsed = parse_vnymr_string(vnymr_str)
            if parsed:
                parsed['time'] = float(row.iloc[0]) + float(row.iloc[1]) * 1e-9
                data_list.append(parsed)
    
    if not data_list:
        raise ValueError(f"No valid VNYMR data found in {filepath}")
    
    result = pd.DataFrame(data_list)
    result.set_index('time', inplace=True)
    return result

def load_gps_data(filepath):
    """Load GPS data from CSV file"""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, header=None, usecols=[0,1,3,4,5,6,7])
    df.columns = ['time_sec', 'time_nsec', 'lat', 'lon', 'alt', 'utm_e', 'utm_n']
    df['time'] = df['time_sec'] + df['time_nsec'] * 1e-9
    df.set_index('time', inplace=True)
    return df[['lat', 'lon', 'alt', 'utm_e', 'utm_n']]

def calibrate_magnetometer(mag_data):
    """
    Calibrate magnetometer for hard and soft iron effects
    Returns: bias (hard iron), scale factors (soft iron)
    """
    # Hard-iron correction: center the data
    bias = mag_data.mean()
    centered = mag_data - bias
    
    # Soft-iron correction: normalize to sphere
    ranges = centered.max() - centered.min()
    avg_range = ranges.mean()
    scale_factors = avg_range / ranges
    
    return bias, scale_factors

def apply_calibration(mag_data, bias, scale_factors):
    """Apply magnetometer calibration"""
    return (mag_data - bias) * scale_factors

def calculate_yaw_from_mag(mag_x, mag_y):
    """Calculate yaw angle from magnetometer data"""
    return np.unwrap(np.arctan2(-mag_y, mag_x))

def butter_filter(data, cutoff, fs, btype='low', order=5):
    """Apply Butterworth filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y

def complementary_filter(mag_yaw, gyro_yaw, alpha=0.98):
    """
    Combine magnetometer and gyro yaw using complementary filter
    alpha: weight for gyro (typically 0.95-0.99)
    """
    filtered = np.zeros_like(mag_yaw)
    filtered[0] = mag_yaw[0]
    
    for i in range(1, len(mag_yaw)):
        # High-pass filtered gyro + low-pass filtered mag
        gyro_delta = gyro_yaw[i] - gyro_yaw[i-1]
        filtered[i] = alpha * (filtered[i-1] + gyro_delta) + (1 - alpha) * mag_yaw[i]
    
    return filtered

# Main analysis
print("="*60)
print("EECE 5554 Lab 5: NUance Navigation Analysis")
print("="*60)

# Load data
imu_circles = load_imu_data('../data/imugoing_in_circles1.csv')
imu_driving = load_imu_data('../data/imudriving1.csv')
gps_circles = load_gps_data('../data/gpsgoing_in_circles1.csv')
gps_driving = load_gps_data('../data/gpsdriving1.csv')

print(f"\nData loaded:")
print(f"  IMU Circles: {len(imu_circles)} samples")
print(f"  IMU Driving: {len(imu_driving)} samples")
print(f"  GPS Circles: {len(gps_circles)} samples")
print(f"  GPS Driving: {len(gps_driving)} samples")

# Step 1: Magnetometer Calibration
print("\n" + "="*60)
print("Step 1: Magnetometer Calibration")
print("="*60)

mag_cal = imu_circles[['mag_x', 'mag_y', 'mag_z']]
bias, scale_factors = calibrate_magnetometer(mag_cal)

print(f"\nHard-iron bias: X={bias['mag_x']:.4f}, Y={bias['mag_y']:.4f}, Z={bias['mag_z']:.4f} µT")
print(f"Soft-iron scale: X={scale_factors['mag_x']:.4f}, Y={scale_factors['mag_y']:.4f}, Z={scale_factors['mag_z']:.4f}")

# Plot 0: Magnetometer calibration
mag_calibrated = apply_calibration(mag_cal, bias, scale_factors)

fig0 = go.Figure()
fig0.add_trace(go.Scatter(x=mag_cal['mag_x'], y=mag_cal['mag_y'], 
                          mode='markers', name='Uncalibrated', marker=dict(size=4)))
fig0.add_trace(go.Scatter(x=mag_calibrated['mag_x'], y=mag_calibrated['mag_y'], 
                          mode='markers', name='Calibrated', marker=dict(size=4)))
fig0.update_layout(title='Plot 0: Magnetometer Calibration (XY Plane)',
                   xaxis_title='Mag X (µT)', yaxis_title='Mag Y (µT)',
                   width=800, height=600)
fig0.write_html('../docs/plot0_mag_calibration.html')
pio.write_image(fig0, '../docs/plot0_mag_calibration.png', width=1200, height=600)
print("\nPlot 0 saved: ../docs/plot0_mag_calibration.html")
print("Plot 0 saved: ../docs/plot0_mag_calibration.png")

# Apply calibration to driving data
mag_driving = imu_driving[['mag_x', 'mag_y', 'mag_z']]
mag_driving_cal = apply_calibration(mag_driving, bias, scale_factors)

# Calculate yaw angles
yaw_uncal = calculate_yaw_from_mag(mag_driving['mag_x'], mag_driving['mag_y'])
yaw_cal = calculate_yaw_from_mag(mag_driving_cal['mag_x'], mag_driving_cal['mag_y'])

# Plot 1: Yaw before/after calibration
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(yaw_uncal),
                          mode='lines', name='Uncalibrated Yaw'))
fig1.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(yaw_cal),
                          mode='lines', name='Calibrated Yaw'))
fig1.update_layout(title='Plot 1: Magnetometer Yaw Estimation',
                   xaxis_title='Time (s)', yaxis_title='Yaw (degrees)',
                   width=1000, height=500)
fig1.write_html('../docs/plot1_mag_yaw.html')
pio.write_image(fig1, '../docs/plot1_mag_yaw.png', width=1200, height=500)
print("Plot 1 saved: ../docs/plot1_mag_yaw.html")
print("Plot 1 saved: ../docs/plot1_mag_yaw.png")

# Step 2: Gyro Integration and Complementary Filter
print("\n" + "="*60)
print("Step 2: Heading Estimation")
print("="*60)

# Calculate sampling parameters
dt = np.median(np.diff(imu_driving.index))
fs = 1 / dt
print(f"\nSampling rate: {fs:.2f} Hz (dt={dt*1000:.2f} ms)")

# Integrate gyro for yaw
gyro_yaw_rad = cumulative_trapezoid(imu_driving['gyro_z'], imu_driving.index, initial=0)
gyro_yaw_rad += yaw_cal[0]  # Start from same initial angle as magnetometer

# Plot 2: Gyro yaw
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(gyro_yaw_rad),
                          mode='lines', name='Gyro Integrated Yaw'))
fig2.update_layout(title='Plot 2: Gyro Yaw Estimation',
                   xaxis_title='Time (s)', yaxis_title='Yaw (degrees)',
                   width=1000, height=500)
fig2.write_html('../docs/plot2_gyro_yaw.html')
pio.write_image(fig2, '../docs/plot2_gyro_yaw.png', width=1200, height=500)
print("Plot 2 saved: ../docs/plot2_gyro_yaw.html")
print("Plot 2 saved: ../docs/plot2_gyro_yaw.png")

# Complementary filter
cutoff_freq = 0.1  # Hz
mag_yaw_lp = butter_filter(yaw_cal, cutoff_freq, fs, 'low')
gyro_yaw_hp = butter_filter(gyro_yaw_rad, cutoff_freq, fs, 'high')
comp_yaw = mag_yaw_lp + gyro_yaw_hp

print(f"\nComplementary filter cutoff: {cutoff_freq} Hz")

# Plot 3: Filter outputs
fig3 = make_subplots(rows=2, cols=2, subplot_titles=(
    'Low-Pass Filtered Magnetometer',
    'High-Pass Filtered Gyro',
    'Complementary Filter Output',
    'Comparison: All Methods'
))

fig3.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(mag_yaw_lp), name='Mag LP'),
               row=1, col=1)
fig3.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(gyro_yaw_hp), name='Gyro HP'),
               row=1, col=2)
fig3.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(comp_yaw), name='Complementary'),
               row=2, col=1)
fig3.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(yaw_cal), name='Mag', line=dict(dash='dash')),
               row=2, col=2)
fig3.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(gyro_yaw_rad), name='Gyro', line=dict(dash='dot')),
               row=2, col=2)
fig3.add_trace(go.Scatter(x=imu_driving.index, y=np.rad2deg(comp_yaw), name='Fused'),
               row=2, col=2)

fig3.update_xaxes(title_text="Time (s)")
fig3.update_yaxes(title_text="Yaw (degrees)")
fig3.update_layout(height=800, width=1200, title_text="Plot 3: Complementary Filter Analysis")
fig3.write_html('../docs/plot3_complementary_filter.html')
pio.write_image(fig3, '../docs/plot3_complementary_filter.png', width=1200, height=800)
print("Plot 3 saved: ../docs/plot3_complementary_filter.html")
print("Plot 3 saved: ../docs/plot3_complementary_filter.png")

# Step 3: Velocity Estimation
print("\n" + "="*60)
print("Step 3: Velocity Estimation")
print("="*60)

# Velocity from accelerometer
accel_forward = imu_driving['accel_x'] - imu_driving['accel_x'].mean()  # Remove bias
vel_accel = cumulative_trapezoid(accel_forward, imu_driving.index, initial=0)

print(f"\nAccel velocity range: {vel_accel.min():.2f} to {vel_accel.max():.2f} m/s")

# Plot 4: Velocity from accelerometer
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=imu_driving.index, y=vel_accel,
                          mode='lines', name='Velocity from Accel'))
fig4.update_layout(title='Plot 4: Forward Velocity from Accelerometer',
                   xaxis_title='Time (s)', yaxis_title='Velocity (m/s)',
                   width=1000, height=500)
fig4.write_html('../docs/plot4_accel_velocity.html')
pio.write_image(fig4, '../docs/plot4_accel_velocity.png', width=1200, height=500)
print("Plot 4 saved: ../docs/plot4_accel_velocity.html")
print("Plot 4 saved: ../docs/plot4_accel_velocity.png")

# Velocity from GPS
dx = gps_driving['utm_e'].diff()
dy = gps_driving['utm_n'].diff()
dt_gps = gps_driving.index.to_series().diff()
vel_gps = np.sqrt(dx**2 + dy**2) / dt_gps

print(f"GPS velocity range: {vel_gps.min():.2f} to {vel_gps.max():.2f} m/s")

# Plot 5: Velocity from GPS
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=gps_driving.index[1:], y=vel_gps[1:],
                          mode='lines', name='Velocity from GPS'))
fig5.update_layout(title='Plot 5: Forward Velocity from GPS',
                   xaxis_title='Time (s)', yaxis_title='Velocity (m/s)',
                   width=1000, height=500)
fig5.write_html('../docs/plot5_gps_velocity.html')
pio.write_image(fig5, '../docs/plot5_gps_velocity.png', width=1200, height=500)
print("Plot 5 saved: ../docs/plot5_gps_velocity.html")
print("Plot 5 saved: ../docs/plot5_gps_velocity.png")

# Step 4: Trajectory Estimation
print("\n" + "="*60)
print("Step 4: Trajectory Estimation")
print("="*60)

# Use complementary filter yaw for trajectory
vx_imu = vel_accel * np.cos(comp_yaw)
vy_imu = vel_accel * np.sin(comp_yaw)

x_imu = cumulative_trapezoid(vx_imu, imu_driving.index, initial=0)
y_imu = cumulative_trapezoid(vy_imu, imu_driving.index, initial=0)

# GPS trajectory (relative to start)
x_gps = gps_driving['utm_e'] - gps_driving['utm_e'].iloc[0]
y_gps = gps_driving['utm_n'] - gps_driving['utm_n'].iloc[0]

print(f"\nIMU trajectory extent: {x_imu.max()-x_imu.min():.1f} x {y_imu.max()-y_imu.min():.1f} m")
print(f"GPS trajectory extent: {x_gps.max()-x_gps.min():.1f} x {y_gps.max()-y_gps.min():.1f} m")

# Plot 6: Trajectory comparison
fig6 = make_subplots(rows=1, cols=2, subplot_titles=('IMU Trajectory', 'GPS Trajectory'))

fig6.add_trace(go.Scatter(x=x_imu, y=y_imu, mode='lines', name='IMU',
                          line=dict(color='blue')), row=1, col=1)
fig6.add_trace(go.Scatter(x=x_gps, y=y_gps, mode='lines', name='GPS',
                          line=dict(color='red')), row=1, col=2)

fig6.update_xaxes(title_text="East (m)", row=1, col=1)
fig6.update_xaxes(title_text="East (m)", row=1, col=2)
fig6.update_yaxes(title_text="North (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
fig6.update_yaxes(title_text="North (m)", scaleanchor="x2", scaleratio=1, row=1, col=2)
fig6.update_layout(height=600, width=1400, title_text="Plot 6: Trajectory Comparison")
fig6.write_html('../docs/plot6_trajectory_comparison.html')
pio.write_image(fig6, '../docs/plot6_trajectory_comparison.png', width=1400, height=600)
print("Plot 6 saved: ../docs/plot6_trajectory_comparison.html")
print("Plot 6 saved: ../docs/plot6_trajectory_comparison.png")

print("\n" + "="*60)
print("Analysis Complete!")
print("All plots saved to ../docs/ directory")
print("="*60)
