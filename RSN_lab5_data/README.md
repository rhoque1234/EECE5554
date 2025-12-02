# EECE 5554 Lab 5: NUance Navigation

## Repository Structure

```
RSN_lab5_data/
├── README.md                    # This file
├── launch/                      # ROS2 launch files
│   └── gps_imu_launch.py       # Launch file for GPS and IMU drivers
├── data/                        # Collected datasets
│   ├── gpsdriving1.csv         # GPS data from driving route
│   ├── imudriving1.csv         # IMU data from driving route
│   ├── gpsgoing_in_circles1.csv # GPS data for calibration
│   ├── imugoing_in_circles1.csv # IMU data for calibration
│   ├── driving1/               # Rosbag for driving data
│   │   ├── driving1_0.mcap
│   │   └── metadata.yaml
│   └── going_in_circles1/      # Rosbag for calibration data
│       ├── going_in_circles1_0.mcap
│       └── metadata.yaml
├── analysis/                    # Analysis notebooks and scripts
│   └── lab5_analysis.ipynb     # Main analysis notebook
├── drivers/                     # Driver source code (if applicable)
└── docs/                        # Documentation and reports

```

## Lab Overview

This lab implements inertial odometry to determine the path of a moving car using:
- **Hardware**: Vectornav VN-100 IMU, Standalone GPS puck
- **Objective**: Calculate magnetometer calibration, integrate gyro/accelerometer data, and compare vehicle path estimates from IMU and GPS

## Getting Started

### Prerequisites
- Python 3.8+
- ROS2 (for data collection)
- Required Python packages: pandas, numpy, scipy, plotly

### Installation
```bash
pip install pandas numpy scipy plotly
```

### Running the Analysis
1. Navigate to the analysis directory:
   ```bash
   cd analysis/
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook lab5_analysis.ipynb
   ```

3. Run all cells to generate the required plots (0-6) and analysis

## Data Collection

### Calibration Data (Going in Circles)
- Location: Ruggles circle near Centennial common
- Duration: 4-5 complete circles
- Purpose: Magnetometer hard and soft iron calibration

### Driving Data
- Distance: 2-3 kilometers
- Minimum turns: 10
- Start/End: Same location (closed loop)
- Constraints: No tunnels (to maintain GPS signal)

## Analysis Components

### Plots Generated
- **Plot 0**: Magnetometer data before and after correction
- **Plot 1**: Magnetometer yaw estimation (uncalibrated vs calibrated)
- **Plot 2**: Gyro yaw estimation vs. time
- **Plot 3**: Filter outputs (low-pass mag, high-pass gyro, complementary filter)
- **Plot 4**: Forward velocity from accelerometer
- **Plot 5**: Forward velocity from GPS
- **Plot 6**: Trajectory comparison (GPS vs IMU)

### Key Algorithms
1. **Magnetometer Calibration**: Hard and soft iron correction using least squares
2. **Heading Estimation**: Complementary filter combining magnetometer and gyro data
3. **Velocity Estimation**: Integration of accelerometer data with adjustments
4. **Trajectory Estimation**: Dead reckoning using velocity and heading

## Launch File

The `launch/gps_imu_launch.py` file launches both GPS and IMU driver nodes simultaneously for synchronized data collection.

## Questions Addressed

The analysis answers 8 questions (Q0-Q7) covering:
- Magnetometer calibration methodology
- Complementary filter design
- Yaw estimate reliability
- Velocity adjustments and discrepancies
- Motion model validation
- Trajectory comparison with GPS
- Position accuracy over time

## Authors

Team members who contributed to this lab.

## License

This project is part of EECE 5554 coursework at Northeastern University.
