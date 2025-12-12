# RTK Data Files

This directory contains processed RTK GNSS data files.

## Files

- `occluded_rtk_data.csv`: Stationary data from occluded location
- `open_rtk_data.csv`: Stationary data from open sky location
- `walking_rtk_data.csv`: Walking data along a path

## Data Format

Each CSV file contains the following fields:

- `utc_time`: UTC timestamp from GNGGA sentence
- `latitude`: Latitude in decimal degrees
- `longitude`: Longitude in decimal degrees
- `altitude`: Altitude in meters (MSL)
- `easting`: UTM Easting coordinate (meters)
- `northing`: UTM Northing coordinate (meters)
- `zone_num`: UTM zone number
- `zone_letter`: UTM zone letter
- `fix_quality`: GNSS fix quality (0=invalid, 1=GPS, 2=DGPS, 4=RTK fixed, 5=RTK float)
- `hdop`: Horizontal Dilution of Precision
- `num_sats`: Number of satellites used
- `gngga_string`: Original GNGGA sentence

## Usage

These files can be used for:
1. Direct analysis in Python/MATLAB
2. Conversion to ROS 2 bag files
3. Visualization and error analysis
