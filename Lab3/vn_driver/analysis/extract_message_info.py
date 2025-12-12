#!/usr/bin/env python3
"""
Extract message definitions from rosbag to understand custom message structure
"""

from rosbags.rosbag2 import Reader
from pathlib import Path

def extract_message_info(bag_path):
    """Extract message type information from rosbag"""
    bag_path = Path(bag_path)
    
    with Reader(bag_path) as reader:
        print(f"Bag: {bag_path}")
        print(f"\nTopics:")
        for connection in reader.connections:
            print(f"  Topic: {connection.topic}")
            print(f"  Type: {connection.msgtype}")
            print(f"  Format: {connection.serialization_format}")
            if hasattr(connection, 'msgdef'):
                print(f"  Definition: {connection.msgdef}")
            print()
        
        # Try to read first message
        print("\nFirst message sample:")
        for connection, timestamp, rawdata in reader.messages():
            print(f"  Timestamp: {timestamp}")
            print(f"  Data length: {len(rawdata)} bytes")
            print(f"  First 100 bytes (hex): {rawdata[:100].hex()}")
            break

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        extract_message_info(sys.argv[1])
    else:
        print("Usage: python extract_message_info.py <bag_path>")
