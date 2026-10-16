"""
Script to fix numpy/pandas compatibility issues
Run this if you encounter "numpy.dtype size changed" errors
"""

import subprocess
import sys

def fix_dependencies():
    """Reinstall numpy and pandas with compatible versions"""
    print("Fixing numpy/pandas compatibility issue...")
    print("="*70)
    
    print("\n1. Uninstalling numpy and pandas...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy", "pandas"])
    
    print("\n2. Installing compatible versions...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.24.0,<2.0.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas>=2.0.0,<3.0.0"])
    
    print("\n" + "="*70)
    print("✓ Dependencies fixed! Please try running your script again.")
    print("="*70)

if __name__ == "__main__":
    fix_dependencies()

