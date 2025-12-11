# Lab 4: Introduction to Inertial Odometry

## Overview
This lab explores the challenges of dead reckoning using IMU data from walking patterns. The goal is to understand the strengths and weaknesses of inertial navigation by attempting to reconstruct walking paths from accelerometer, gyroscope, and magnetometer data.

## Data Collection
- **Circle Walking**: 60 seconds of walking in a 5m radius circle at 1.2 m/s
- **Square Walking**: 60 seconds of walking in a 5m square pattern with 90-degree turns

## Generated Plots

### Circle Walking (Figures 1-8)
- **Figure 1**: Magnetometer X vs Y calibration (before/after hard iron correction)
- **Figure 2**: Gyroscope X-axis (rate and integrated rotation)
- **Figure 3**: Gyroscope Y-axis (rate and integrated rotation)  
- **Figure 4**: Gyroscope Z-axis, integrated rotation, and magnetometer heading
- **Figure 5**: Accelerometer X-axis (acceleration and integrated velocity)
- **Figure 6**: Accelerometer Y-axis (acceleration and integrated velocity)
- **Figure 7**: Accelerometer Z-axis (acceleration and integrated velocity)
- **Figure 8**: North vs East trajectory (magnetometer vs gyro heading)

### Square Walking (Figures 9-16)
- **Figure 9**: Magnetometer X vs Y with calibration applied
- **Figure 10**: Gyroscope X-axis analysis
- **Figure 11**: Gyroscope Y-axis analysis
- **Figure 12**: Gyroscope Z-axis and heading analysis
- **Figure 13**: Accelerometer X-axis analysis
- **Figure 14**: Accelerometer Y-axis analysis
- **Figure 15**: Accelerometer Z-axis analysis
- **Figure 16**: North vs East trajectory estimate

## Key Observations

### Results Analysis
The trajectory reconstructions show significant drift and error accumulation, which is expected with pure inertial navigation. Key observations include:

1. **Magnetometer Calibration**: Hard iron offsets were successfully removed, showing circular patterns before and after calibration
2. **Gyroscope Integration**: Heading estimates from gyroscope integration show continuous drift over time due to bias and noise
3. **Accelerometer Double Integration**: Position estimates from twice-integrated acceleration exhibit severe drift, making long-term position estimation unreliable
4. **Heading Comparison**: Magnetometer-based heading is more stable than gyroscope integration alone, but still shows noise

### Ground Truth Comparison
Given the simulated nature of this data (no physical video available):
- Circle walking should show a circular trajectory with return to origin
- Square walking should show 90-degree turns and straight segments
- Actual results show drift and position errors accumulating over time
- This demonstrates why IMU-only navigation is insufficient for accurate positioning

### Plans for Lab 5 (Vehicle Application)
For improved results in Lab 5:

1. **Sensor Fusion**: Combine IMU with GPS for absolute position corrections
2. **Complementary Filtering**: Use gyroscope for short-term orientation and magnetometer for long-term correction
3. **Bias Estimation**: Implement online bias estimation and removal
4. **Zero Velocity Updates**: Use vehicle stops to reset velocity drift
5. **Mounting**: Ensure IMU is rigidly mounted and aligned with vehicle axes
6. **Sampling Rate**: Verify 40Hz sampling is sufficient for vehicle dynamics

## Repository Structure
```
Lab4/
├── data/
│   ├── circle/         # Circle walking IMU data (numpy format)
│   └── square/         # Square walking IMU data (numpy format)
├── analysis/
│   ├── generate_walking_data.py    # Data generation script
│   ├── analyze_circle.py           # Circle analysis (Figs 1-8)
│   └── analyze_square.py           # Square analysis (Figs 9-16)
└── results/
    ├── circle/         # Figures 1-8
    └── square/         # Figures 9-16
```

## Running the Analysis

### Generate Data:
```bash
python Lab4/analysis/generate_walking_data.py --mode circle --duration 60 --output Lab4/data/circle
python Lab4/analysis/generate_walking_data.py --mode square --duration 60 --output Lab4/data/square
```

### Generate Plots:
```bash
python Lab4/analysis/analyze_circle.py --data Lab4/data/circle --output Lab4/results/circle
python Lab4/analysis/analyze_square.py --data Lab4/data/square --output Lab4/results/square
```

## Technical Details

### IMU Specifications (VectorNav VN-100)
- Sample Rate: 40 Hz
- Gyroscope Range: ±2000 deg/s
- Accelerometer Range: ±16g
- Magnetometer: 3-axis, Earth's magnetic field measurement

### Data Processing
- **Magnetometer Calibration**: Hard iron offset removal using min/max method
- **Integration Method**: Cumulative trapezoidal integration
- **Coordinate Frames**: Body frame (IMU) → World frame (North-East-Down)
- **Heading**: Calculated using atan2(mag_y, mag_x) after calibration

## Conclusions
This lab clearly demonstrates that pure inertial navigation is extremely challenging due to:
- Sensor noise and bias
- Error accumulation through integration
- Lack of absolute position reference

These results motivate the need for sensor fusion approaches (IMU + GPS) in Lab 5.
