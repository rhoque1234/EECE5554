#!/usr/bin/env python3


import socket
import time
import math
import random
import argparse
import numpy as np


class IMUEmulator:
    def __init__(self, port=5555, mode='stationary', duration=300):
        """
        Initialize IMU emulator
        
        Args:
            port: TCP port to listen on (emulates serial port)
            mode: 'stationary' or 'motion' 
            duration: data collection duration in seconds
        """
        self.port = port
        self.mode = mode
        self.duration = duration
        self.rate = 40  # Hz - VectorNav default after configuration
        
        # Noise parameters (realistic for VN-100)
        # Gyro noise (deg/s)
        self.gyro_noise_std = 0.0035  # degrees/s (angle random walk)
        self.gyro_bias = np.array([0.05, -0.03, 0.02])  # deg/s
        self.gyro_bias_instability = 0.01  # deg/s
        
        # Accel noise (m/s^2)
        self.accel_noise_std = 0.0014  # m/s^2
        self.accel_bias = np.array([0.01, -0.02, 0.015])  # m/s^2
        
        # Mag noise (Gauss)
        self.mag_noise_std = 0.0015  # Gauss
        
        # Initial orientation (roll, pitch, yaw in degrees)
        self.orientation = np.array([0.0, 0.0, 0.0])
        
        # Initialize biases with random walk
        self.current_gyro_bias = self.gyro_bias.copy()
        self.current_accel_bias = self.accel_bias.copy()
        
    def generate_stationary_data(self):
        """Generate stationary IMU data with realistic noise"""
        # Gyro: near zero with noise and bias drift
        gyro = self.current_gyro_bias + np.random.normal(0, self.gyro_noise_std, 3)
        
        # Add slow bias drift (random walk)
        self.current_gyro_bias += np.random.normal(0, 0.0001, 3)
        
        # Accel: gravity on Z axis with noise
        accel = np.array([0.0, 0.0, 9.81]) + self.current_accel_bias + np.random.normal(0, self.accel_noise_std, 3)
        
        # Magnetometer: roughly pointing north with noise
        mag = np.array([0.3, 0.0, 0.4]) + np.random.normal(0, self.mag_noise_std, 3)
        
        # Orientation stays mostly constant with tiny drift
        self.orientation += np.random.normal(0, 0.001, 3)
        
        return gyro, accel, self.orientation.copy(), mag
    
    def generate_motion_data(self, t):
        """Generate motion IMU data with interesting movements"""
        # Create a sinusoidal motion pattern
        freq = 0.5  # Hz
        
        # Gyro: rotational rates
        gyro_x = 10 * math.sin(2 * math.pi * freq * t) + np.random.normal(0, self.gyro_noise_std)
        gyro_y = 5 * math.cos(2 * math.pi * freq * 0.7 * t) + np.random.normal(0, self.gyro_noise_std)
        gyro_z = 8 * math.sin(2 * math.pi * freq * 1.2 * t) + np.random.normal(0, self.gyro_noise_std)
        gyro = np.array([gyro_x, gyro_y, gyro_z])
        
        # Integrate gyro to get orientation
        dt = 1.0 / self.rate
        self.orientation += gyro * dt
        
        # Accel: gravity + motion
        accel_motion = np.array([
            2.0 * math.sin(2 * math.pi * freq * 2 * t),
            1.5 * math.cos(2 * math.pi * freq * 1.5 * t),
            0.5 * math.sin(2 * math.pi * freq * 3 * t)
        ])
        accel = np.array([0.0, 0.0, 9.81]) + accel_motion + np.random.normal(0, self.accel_noise_std, 3)
        
        # Magnetometer changes slightly with orientation
        mag = np.array([
            0.3 + 0.05 * math.sin(self.orientation[0] * math.pi / 180),
            0.0 + 0.05 * math.cos(self.orientation[1] * math.pi / 180),
            0.4 + 0.03 * math.sin(self.orientation[2] * math.pi / 180)
        ]) + np.random.normal(0, self.mag_noise_std, 3)
        
        return gyro, accel, self.orientation.copy(), mag
    
    def create_vnymr_string(self, yaw, pitch, roll, mag_x, mag_y, mag_z, 
                           accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
        """
        Create VNYMR string matching VectorNav format
        Format: $VNYMR,yaw,pitch,roll,mag_x,mag_y,mag_z,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z*checksum
        """
        # Create data string without checksum
        data = f"VNYMR,{yaw:.2f},{pitch:.2f},{roll:.2f},{mag_x:.4f},{mag_y:.4f},{mag_z:.4f}," \
               f"{accel_x:.4f},{accel_y:.4f},{accel_z:.4f},{gyro_x:.4f},{gyro_y:.4f},{gyro_z:.4f}"
        
        # Calculate checksum (XOR of all bytes between $ and *)
        checksum = 0
        for char in data:
            checksum ^= ord(char)
        
        # Create full NMEA string
        nmea_string = f"${data}*{checksum:02X}\r\n"
        return nmea_string
    
    def run(self):
        """Run the IMU data emulator"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.port))
        server_socket.listen(1)
        
        print(f"VectorNav VN-100 emulator listening on localhost:{self.port}")
        print(f"Mode: {self.mode}, Duration: {self.duration}s, Rate: {self.rate}Hz")
        print("Waiting for connection...")
        
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        
        start_time = time.time()
        sample_count = 0
        
        try:
            # Listen for configuration commands
            conn.settimeout(0.1)
            
            while time.time() - start_time < self.duration:
                # Check for incoming configuration
                try:
                    data = conn.recv(1024)
                    if data:
                        print(f"Received configuration command: {data.hex()}")
                except socket.timeout:
                    pass
                
                # Generate and send IMU data
                t = time.time() - start_time
                
                if self.mode == 'stationary':
                    gyro, accel, orientation, mag = self.generate_stationary_data()
                else:
                    gyro, accel, orientation, mag = self.generate_motion_data(t)
                
                # Create VNYMR string
                # VectorNav outputs: yaw, pitch, roll (degrees), mag (Gauss), accel (m/s^2), gyro (deg/s)
                vnymr = self.create_vnymr_string(
                    orientation[2], orientation[1], orientation[0],  # yaw, pitch, roll
                    mag[0], mag[1], mag[2],
                    accel[0], accel[1], accel[2],
                    gyro[0], gyro[1], gyro[2]
                )
                
                try:
                    conn.sendall(vnymr.encode('utf-8'))
                    sample_count += 1
                except BrokenPipeError:
                    print("Connection closed by client")
                    break
                
                # Maintain sample rate
                time.sleep(1.0 / self.rate)
                
                if sample_count % 40 == 0:  # Print every second
                    print(f"Time: {t:.1f}s, Samples: {sample_count}")
        
        except KeyboardInterrupt:
            print("\nData collection stopped by user")
        finally:
            conn.close()
            server_socket.close()
            print(f"Data collection complete. Total samples: {sample_count}")


def main():
    parser = argparse.ArgumentParser(description='VectorNav IMU Emulator')
    parser.add_argument('--port', type=int, default=5555, help='TCP port to listen on')
    parser.add_argument('--mode', choices=['stationary', 'motion'], default='stationary',
                       help='Collection mode: stationary or motion')
    parser.add_argument('--duration', type=int, default=300,
                       help='Collection duration in seconds (default: 300 = 5 minutes)')
    
    args = parser.parse_args()
    
    emulator = IMUEmulator(port=args.port, mode=args.mode, duration=args.duration)
    emulator.run()


if __name__ == '__main__':
    main()
