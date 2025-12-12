#!/usr/bin/env python3
"""
Complete Lab 2 workflow script
Processes RTK data files directly and generates all analysis
"""

import subprocess
import sys
import time
from pathlib import Path

# Check and install dependencies
print("Checking dependencies...")
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import utm
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "matplotlib", "utm"])
    import numpy as np
    import matplotlib.pyplot as plt
    import utm

print("All dependencies installed!\n")

# Now run the actual analysis
print("="*70)
print("LAB 2 RTK GNSS ANALYSIS")
print("="*70)

# Import and run the analysis
sys.path.insert(0, str(Path(__file__).parent))
from rtk_analysis import main as run_analysis

print("\nStarting analysis...\n")
run_analysis()

print("\n" + "="*70)
print("LAB 2 ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  - rtk_stationary_ne.png")
print("  - rtk_stationary_altitude.png")
print("  - rtk_stationary_histograms.png")
print("  - rtk_walking_ne.png")
print("  - rtk_walking_altitude.png")
print("\nNext steps:")
print("  1. Review all generated plots")
print("  2. Compare with Lab 1 standalone GPS results")
print("  3. Use error values in your lab report")
print("  4. Answer discussion questions")
print("="*70)
