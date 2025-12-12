#!/usr/bin/env python3
"""
Analysis script for RTK GNSS data
Generates all required plots and error calculations for Lab 2
"""

import numpy as np
import matplotlib.pyplot as plt
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
        
        # Fix quality
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
            'zone_letter': zone_letter
        }
    except (ValueError, IndexError) as e:
        return None


def load_rtk_data(filepath):
    """Load and parse RTK data from a text file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parsed = parse_gngga(line)
            if parsed:
                data.append(parsed)
    return data


def extract_arrays(data):
    """Extract numpy arrays from parsed data"""
    eastings = np.array([d['easting'] for d in data])
    northings = np.array([d['northing'] for d in data])
    altitudes = np.array([d['altitude'] for d in data])
    hdops = np.array([d['hdop'] for d in data])
    fix_qualities = np.array([d['fix_quality'] for d in data])
    return eastings, northings, altitudes, hdops, fix_qualities


def plot_stationary_ne(occluded_e, occluded_n, open_e, open_n, output_dir):
    """Plot stationary northing vs easting (centered on centroid)"""
    # Calculate centroids
    occluded_centroid_e = np.mean(occluded_e)
    occluded_centroid_n = np.mean(occluded_n)
    open_centroid_e = np.mean(open_e)
    open_centroid_n = np.mean(open_n)
    
    # Subtract centroids
    occluded_e_centered = occluded_e - occluded_centroid_e
    occluded_n_centered = occluded_n - occluded_centroid_n
    open_e_centered = open_e - open_centroid_e
    open_n_centered = open_n - open_centroid_n
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Occluded subplot
    ax1.scatter(occluded_e_centered, occluded_n_centered, alpha=0.6, s=20, c='red', label='Occluded RTK')
    ax1.scatter(0, 0, c='black', s=100, marker='x', linewidths=3, label='Centroid')
    ax1.set_xlabel('Easting - Centroid (m)', fontsize=12)
    ax1.set_ylabel('Northing - Centroid (m)', fontsize=12)
    ax1.set_title('Stationary RTK: Occluded Location', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    ax1.text(0.02, 0.98, f'Centroid Easting: {occluded_centroid_e:.2f} m\nCentroid Northing: {occluded_centroid_n:.2f} m',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Open subplot
    ax2.scatter(open_e_centered, open_n_centered, alpha=0.6, s=20, c='blue', label='Open RTK')
    ax2.scatter(0, 0, c='black', s=100, marker='x', linewidths=3, label='Centroid')
    ax2.set_xlabel('Easting - Centroid (m)', fontsize=12)
    ax2.set_ylabel('Northing - Centroid (m)', fontsize=12)
    ax2.set_title('Stationary RTK: Open Location', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    ax2.text(0.02, 0.98, f'Centroid Easting: {open_centroid_e:.2f} m\nCentroid Northing: {open_centroid_n:.2f} m',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rtk_stationary_ne.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return occluded_e_centered, occluded_n_centered, open_e_centered, open_n_centered


def plot_stationary_altitude(occluded_alt, open_alt, output_dir):
    """Plot stationary altitude vs time"""
    occluded_time = np.arange(len(occluded_alt))
    open_time = np.arange(len(open_alt))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Occluded subplot
    ax1.plot(occluded_time, occluded_alt, 'r-', alpha=0.7, linewidth=1, label='Occluded RTK')
    ax1.axhline(y=np.mean(occluded_alt), color='black', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(occluded_alt):.2f} m')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title('Stationary RTK Altitude: Occluded Location', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.text(0.02, 0.98, f'Std Dev: {np.std(occluded_alt):.3f} m\nMin: {np.min(occluded_alt):.2f} m\nMax: {np.max(occluded_alt):.2f} m',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Open subplot
    ax2.plot(open_time, open_alt, 'b-', alpha=0.7, linewidth=1, label='Open RTK')
    ax2.axhline(y=np.mean(open_alt), color='black', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(open_alt):.2f} m')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Altitude (m)', fontsize=12)
    ax2.set_title('Stationary RTK Altitude: Open Location', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.text(0.02, 0.98, f'Std Dev: {np.std(open_alt):.3f} m\nMin: {np.min(open_alt):.2f} m\nMax: {np.max(open_alt):.2f} m',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rtk_stationary_altitude.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_stationary_histograms(occluded_e_centered, occluded_n_centered, 
                                open_e_centered, open_n_centered, output_dir):
    """Plot histograms of distance from centroid"""
    # Calculate Euclidean distances from centroid
    occluded_distances = np.sqrt(occluded_e_centered**2 + occluded_n_centered**2)
    open_distances = np.sqrt(open_e_centered**2 + open_n_centered**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Occluded histogram
    ax1.hist(occluded_distances, bins=30, color='red', alpha=0.7, edgecolor='black')
    ax1.axvline(x=np.mean(occluded_distances), color='darkred', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(occluded_distances):.3f} m')
    ax1.axvline(x=np.median(occluded_distances), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(occluded_distances):.3f} m')
    ax1.set_xlabel('Distance from Centroid (m)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Stationary RTK: Occluded Location\nDistance Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    ax1.text(0.98, 0.98, f'Std Dev: {np.std(occluded_distances):.3f} m\nRMS: {np.sqrt(np.mean(occluded_distances**2)):.3f} m',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Open histogram
    ax2.hist(open_distances, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(open_distances), color='darkblue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(open_distances):.3f} m')
    ax2.axvline(x=np.median(open_distances), color='cyan', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(open_distances):.3f} m')
    ax2.set_xlabel('Distance from Centroid (m)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Stationary RTK: Open Location\nDistance Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    ax2.text(0.98, 0.98, f'Std Dev: {np.std(open_distances):.3f} m\nRMS: {np.sqrt(np.mean(open_distances**2)):.3f} m',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rtk_stationary_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return occluded_distances, open_distances


def plot_walking_ne(walking_e, walking_n, output_dir):
    """Plot walking northing vs easting with line of best fit"""
    # Calculate line of best fit
    coeffs = np.polyfit(walking_e, walking_n, 1)
    poly_func = np.poly1d(coeffs)
    walking_n_fit = poly_func(walking_e)
    
    # Calculate residuals and RMSE
    residuals = walking_n - walking_n_fit
    rmse_walking = np.sqrt(np.mean(residuals**2))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(walking_e, walking_n, c=range(len(walking_e)), cmap='viridis', s=50, alpha=0.7, 
                edgecolors='black', linewidth=0.5, label='Walking Path')
    plt.plot(walking_e, walking_n_fit, 'r-', linewidth=3, 
             label=f'Best Fit: y = {coeffs[0]:.4f}x + {coeffs[1]:.2f}')
    
    # Add start and end markers
    plt.scatter(walking_e[0], walking_n[0], c='green', s=200, marker='o', 
                edgecolors='black', linewidth=2, label='Start', zorder=5)
    plt.scatter(walking_e[-1], walking_n[-1], c='red', s=200, marker='s', 
                edgecolors='black', linewidth=2, label='End', zorder=5)
    
    plt.xlabel('Easting (m)', fontsize=12)
    plt.ylabel('Northing (m)', fontsize=12)
    plt.title('Walking RTK: Northing vs Easting with Best Fit Line', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.colorbar(label='Time Progression', alpha=0.7)
    
    stats_text = f'RMSE from line: {rmse_walking:.3f} m\nSlope: {coeffs[0]:.4f}\nIntercept: {coeffs[1]:.2f} m'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rtk_walking_ne.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rmse_walking, residuals


def plot_walking_altitude(walking_alt, output_dir):
    """Plot walking altitude vs time"""
    walking_time = np.arange(len(walking_alt))
    
    plt.figure(figsize=(12, 6))
    plt.plot(walking_time, walking_alt, 'g-', linewidth=2, alpha=0.7, label='Walking RTK Altitude')
    plt.scatter(walking_time, walking_alt, c=range(len(walking_alt)), cmap='viridis', s=30, alpha=0.6, zorder=5)
    plt.axhline(y=np.mean(walking_alt), color='black', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(walking_alt):.2f} m')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Altitude (m)', fontsize=12)
    plt.title('Walking RTK: Altitude vs Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.colorbar(label='Time Progression')
    
    stats_text = f'Mean: {np.mean(walking_alt):.2f} m\nStd Dev: {np.std(walking_alt):.3f} m\nMin: {np.min(walking_alt):.2f} m\nMax: {np.max(walking_alt):.2f} m'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rtk_walking_altitude.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_errors(occluded_distances, open_distances, occluded_hdop, open_hdop, 
                     walking_rmse, walking_residuals, walking_hdop):
    """Calculate and print error statistics"""
    print("=" * 70)
    print("RTK GNSS ERROR ANALYSIS")
    print("=" * 70)
    
    print("\n### STATIONARY DATA ###\n")
    
    print("Occluded RTK:")
    print(f"  RMS Error: {np.sqrt(np.mean(occluded_distances**2)):.4f} m")
    print(f"  Mean Error: {np.mean(occluded_distances):.4f} m")
    print(f"  Std Dev: {np.std(occluded_distances):.4f} m")
    print(f"  Max Error: {np.max(occluded_distances):.4f} m")
    print(f"  Mean HDOP: {np.mean(occluded_hdop):.2f}")
    print(f"  95% Confidence (2σ): {2 * np.std(occluded_distances):.4f} m")
    
    print("\nOpen RTK:")
    print(f"  RMS Error: {np.sqrt(np.mean(open_distances**2)):.4f} m")
    print(f"  Mean Error: {np.mean(open_distances):.4f} m")
    print(f"  Std Dev: {np.std(open_distances):.4f} m")
    print(f"  Max Error: {np.max(open_distances):.4f} m")
    print(f"  Mean HDOP: {np.mean(open_hdop):.2f}")
    print(f"  95% Confidence (2σ): {2 * np.std(open_distances):.4f} m")
    
    print("\n### WALKING DATA ###\n")
    print("Walking RTK:")
    print(f"  RMSE from best fit line: {walking_rmse:.4f} m")
    print(f"  Mean Absolute Error: {np.mean(np.abs(walking_residuals)):.4f} m")
    print(f"  Std Dev: {np.std(walking_residuals):.4f} m")
    print(f"  Max Deviation: {np.max(np.abs(walking_residuals)):.4f} m")
    print(f"  Mean HDOP: {np.mean(walking_hdop):.2f}")
    
    print("\n" + "=" * 70)


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'dataset'
    output_dir = script_dir
    output_dir.mkdir(exist_ok=True)
    
    print("Loading RTK datasets...")
    occluded_rtk = load_rtk_data(data_dir / 'occludedRTK.txt')
    open_rtk = load_rtk_data(data_dir / 'openRTK.txt')
    walking_rtk = load_rtk_data(data_dir / 'walkingRTK.txt')
    
    print(f"Occluded RTK: {len(occluded_rtk)} data points")
    print(f"Open RTK: {len(open_rtk)} data points")
    print(f"Walking RTK: {len(walking_rtk)} data points\n")
    
    # Extract arrays
    occluded_e, occluded_n, occluded_alt, occluded_hdop, occluded_fq = extract_arrays(occluded_rtk)
    open_e, open_n, open_alt, open_hdop, open_fq = extract_arrays(open_rtk)
    walking_e, walking_n, walking_alt, walking_hdop, walking_fq = extract_arrays(walking_rtk)
    
    # Generate plots
    print("Generating plots...\n")
    
    print("1. Stationary N/E plots...")
    occluded_e_c, occluded_n_c, open_e_c, open_n_c = plot_stationary_ne(
        occluded_e, occluded_n, open_e, open_n, output_dir)
    
    print("2. Stationary altitude plots...")
    plot_stationary_altitude(occluded_alt, open_alt, output_dir)
    
    print("3. Stationary histogram plots...")
    occluded_distances, open_distances = plot_stationary_histograms(
        occluded_e_c, occluded_n_c, open_e_c, open_n_c, output_dir)
    
    print("4. Walking N/E plot...")
    walking_rmse, walking_residuals = plot_walking_ne(walking_e, walking_n, output_dir)
    
    print("5. Walking altitude plot...")
    plot_walking_altitude(walking_alt, output_dir)
    
    print("\nCalculating error metrics...\n")
    calculate_errors(occluded_distances, open_distances, occluded_hdop, open_hdop,
                    walking_rmse, walking_residuals, walking_hdop)
    
    print("\nAll plots saved to:", output_dir.absolute())
    print("Analysis complete!")


if __name__ == '__main__':
    main()
