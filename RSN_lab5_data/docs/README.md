# Lab 5 Analysis Results

This directory contains all the analysis results for EECE 5554 Lab 5: NUance Navigation.

## Generated Files

### Analysis Plots
**Interactive HTML (for detailed viewing)**
- **plot0_mag_calibration.html** - Magnetometer data before and after correction
- **plot1_mag_yaw.html** - Magnetometer yaw estimation (uncalibrated vs calibrated)
- **plot2_gyro_yaw.html** - Gyro yaw estimation over time
- **plot3_complementary_filter.html** - Filter outputs and comparison
- **plot4_accel_velocity.html** - Forward velocity from accelerometer
- **plot5_gps_velocity.html** - Forward velocity from GPS
- **plot6_trajectory_comparison.html** - IMU vs GPS trajectory comparison

**Static PNG Images (for reports/submissions)**
- **plot0_mag_calibration.png** (147 KB)
- **plot1_mag_yaw.png** (206 KB)
- **plot2_gyro_yaw.png** (166 KB)
- **plot3_complementary_filter.png** (350 KB)
- **plot4_accel_velocity.png** (64 KB)
- **plot5_gps_velocity.png** (63 KB)
- **plot6_trajectory_comparison.png** (170 KB)

### Report
- **LAB5_REPORT.md** - Complete analysis report with answers to all 8 questions (Q0-Q7)

## How to View

### HTML Plots (Interactive)
Open any of the HTML files in a web browser to view interactive Plotly plots. You can:
- Zoom in/out
- Pan around
- Hover for detailed values
- Toggle traces on/off

```bash
firefox plot0_mag_calibration.html  # or any browser
```

### PNG Images (Static)
Open with any image viewer or include directly in your report:
```bash
eog plot0_mag_calibration.png  # Eye of GNOME
# or simply open in any image viewer
```

### Report
View LAB5_REPORT.md in any markdown viewer or text editor. It contains:
- Summary of calibration results
- Sensor performance metrics
- Detailed answers to all lab questions
- Analysis of results and error sources

## Key Results

- **Magnetometer Calibration**: Successfully corrected hard-iron and soft-iron distortions
- **Heading Estimation**: Complementary filter (0.1 Hz cutoff) effectively fused mag and gyro
- **Velocity Estimation**: GPS velocity (0-11.7 m/s) much more reliable than accelerometer (-34 to +16 m/s)
- **Trajectory**: IMU trajectory 6.7x larger than GPS due to velocity drift
- **Navigation Duration**: Pure INS accurate for only ~15-30 seconds without GPS updates

## Submission

For Canvas submission:
1. **Include PNG images** for all 7 plots (easy to embed in reports)
2. Submit **LAB5_REPORT.md** as your written report
3. Reference specific plot numbers when answering questions
4. HTML files are available for interactive viewing if needed
