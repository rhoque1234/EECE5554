# EECE5554 - Robotics Sensing and Navigation

## Lab 2: RTK GNSS

This repository contains the RTK GNSS driver and analysis for Lab 2.

### Repository Structure

```
EECE5554/
 gnss/
     src/
        rtk_driver/          # RTK GNSS driver package
        custom_msgs/         # Custom message definitions
     analysis/
        rtk_analysis.py      # Analysis scripts
        create_ros2_bags.py  # Bag file generator
        run_complete_analysis.py
     data/
         occluded_rtk/        # ROS 2 bag files
         open_rtk/
         walking_rtk/
```

### RTK Driver Features

- Parses GNGGA sentences with fix quality and HDOP
- Converts to UTM coordinates
- Publishes custom RTK messages
- Configurable launch file for any serial port

### Data Files

Three ROS 2 bag files are included:
- **occluded_rtk**: Stationary data from occluded location (309 messages)
- **open_rtk**: Stationary data from open location (322 messages)
- **walking_rtk**: Walking data (76 messages)

### Usage

Build the workspace:
```bash
colcon build
source install/setup.bash
```

Launch the driver:
```bash
ros2 launch rtk_driver rtk_driver_launch.py port:=/dev/pts/2
```

Play bag files:
```bash
ros2 bag play data/occluded_rtk
```

Run analysis:
```bash
cd gnss/analysis
python run_complete_analysis.py
```
