#!/usr/bin/env python3
"""
Script to process RTK .txt files and create CSV outputs that simulate ROS bag data
This creates structured data files that contain all the information that would be in a ROS bag
"""

import csv
import utm
from pathlib import Path


def parse_gngga(line):
    """Parse GNGGA sentence and extract key data"""
    if not line.startswith('$GNGGA'):
        return None
    
    parts = line.strip().split(',')
    if len(parts) < 15:
        return None
    
    try:
        # UTC time
        utc_time = parts[1]
        
        # Latitude
        lat_str = parts[2]
        lat_dir = parts[3]
        if lat_str and lat_dir:
            lat_deg = float(lat_str[:2])
            lat_min = float(lat_str[2:])
            latitude = lat_deg + lat_min / 60.0
            if lat_dir == 'S':
                latitude = -latitude
        else:
            return None
        
        # Longitude
        lon_str = parts[4]
        lon_dir = parts[5]
        if lon_str and lon_dir:
            lon_deg = float(lon_str[:3])
            lon_min = float(lon_str[3:])
            longitude = lon_deg + lon_min / 60.0
            if lon_dir == 'W':
                longitude = -longitude
        else:
            return None
        
        # Fix quality (0=invalid, 1=GPS fix, 2=DGPS fix, 4=RTK fixed, 5=RTK float)
        fix_quality = int(parts[6]) if parts[6] else 0
        
        # Number of satellites
        num_sats = int(parts[7]) if parts[7] else 0
        
        # HDOP
        hdop = float(parts[8]) if parts[8] else 999.9
        
        # Altitude
        altitude = float(parts[9]) if parts[9] else 0.0
        
        # Convert to UTM
        utm_data = utm.from_latlon(latitude, longitude)
        easting = utm_data[0]
        northing = utm_data[1]
        zone_num = utm_data[2]
        zone_letter = utm_data[3]
        
        return {
            'utc_time': utc_time,
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'fix_quality': fix_quality,
            'num_sats': num_sats,
            'hdop': hdop,
            'easting': easting,
            'northing': northing,
            'zone_num': zone_num,
            'zone_letter': zone_letter,
            'gngga_string': line.strip()
        }
    except (ValueError, IndexError) as e:
        return None


def process_rtk_file(input_file, output_file):
    """Process RTK .txt file and create CSV output"""
    data_points = []
    
    print(f"\nProcessing {input_file.name}...")
    
    with open(input_file, 'r') as f:
        for line in f:
            parsed = parse_gngga(line)
            if parsed:
                data_points.append(parsed)
    
    print(f"  - Parsed {len(data_points)} valid GNGGA sentences")
    
    # Write to CSV
    if data_points:
        fieldnames = ['utc_time', 'latitude', 'longitude', 'altitude', 
                     'easting', 'northing', 'zone_num', 'zone_letter',
                     'fix_quality', 'hdop', 'num_sats', 'gngga_string']
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_points)
        
        print(f"  - Created {output_file.name}")
        
        # Print statistics
        avg_hdop = sum(d['hdop'] for d in data_points) / len(data_points)
        fix_types = {}
        for d in data_points:
            fix_types[d['fix_quality']] = fix_types.get(d['fix_quality'], 0) + 1
        
        print(f"  - Average HDOP: {avg_hdop:.2f}")
        print(f"  - Fix quality distribution: {fix_types}")
        
        return len(data_points)
    
    return 0


def main():
    """Process all RTK datasets"""
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / 'dataset'
    data_dir = base_dir / 'data'
    
    # Ensure data directory exists
    data_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("RTK Data Processing - Creating Bag-equivalent CSV Files")
    print("="*60)
    
    datasets = [
        ('occludedRTK.txt', 'occluded_rtk_data.csv'),
        ('openRTK.txt', 'open_rtk_data.csv'),
        ('walkingRTK.txt', 'walking_rtk_data.csv')
    ]
    
    total_points = 0
    for input_name, output_name in datasets:
        input_file = dataset_dir / input_name
        output_file = data_dir / output_name
        
        if input_file.exists():
            points = process_rtk_file(input_file, output_file)
            total_points += points
        else:
            print(f"\nWARNING: {input_file} not found!")
    
    print("\n" + "="*60)
    print(f"Processing complete! Total data points: {total_points}")
    print(f"Output files saved to: {data_dir}")
    print("="*60)
    
    # Create a summary file
    summary_file = data_dir / 'README.md'
    with open(summary_file, 'w') as f:
        f.write("# RTK Data Files\n\n")
        f.write("This directory contains processed RTK GNSS data files.\n\n")
        f.write("## Files\n\n")
        f.write("- `occluded_rtk_data.csv`: Stationary data from occluded location\n")
        f.write("- `open_rtk_data.csv`: Stationary data from open sky location\n")
        f.write("- `walking_rtk_data.csv`: Walking data along a path\n\n")
        f.write("## Data Format\n\n")
        f.write("Each CSV file contains the following fields:\n\n")
        f.write("- `utc_time`: UTC timestamp from GNGGA sentence\n")
        f.write("- `latitude`: Latitude in decimal degrees\n")
        f.write("- `longitude`: Longitude in decimal degrees\n")
        f.write("- `altitude`: Altitude in meters (MSL)\n")
        f.write("- `easting`: UTM Easting coordinate (meters)\n")
        f.write("- `northing`: UTM Northing coordinate (meters)\n")
        f.write("- `zone_num`: UTM zone number\n")
        f.write("- `zone_letter`: UTM zone letter\n")
        f.write("- `fix_quality`: GNSS fix quality (0=invalid, 1=GPS, 2=DGPS, 4=RTK fixed, 5=RTK float)\n")
        f.write("- `hdop`: Horizontal Dilution of Precision\n")
        f.write("- `num_sats`: Number of satellites used\n")
        f.write("- `gngga_string`: Original GNGGA sentence\n\n")
        f.write("## Usage\n\n")
        f.write("These files can be used for:\n")
        f.write("1. Direct analysis in Python/MATLAB\n")
        f.write("2. Conversion to ROS 2 bag files\n")
        f.write("3. Visualization and error analysis\n")
    
    print(f"\nCreated {summary_file.name} with data description")


if __name__ == '__main__':
    main()
