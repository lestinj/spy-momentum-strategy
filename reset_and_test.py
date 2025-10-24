#!/usr/bin/env python3
"""
Quick Reset and Test Script for ML-Adaptive V49
"""

import os
import sys

print("="*60)
print("ML-ADAPTIVE V49 - RESET AND TEST")
print("="*60)

# Step 1: Clean old files
print("\n1. Cleaning old configuration files...")
files_to_remove = ['capital_config.json', 'adaptive_positions.json']
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"   ✅ Removed {file}")
    else:
        print(f"   ⏭️  {file} not found (OK)")

if os.path.exists('adaptive_ml_models'):
    import shutil
    shutil.rmtree('adaptive_ml_models')
    print("   ✅ Removed ML models directory")

# Step 2: Set capital
print("\n2. Setting capital to $16,000...")
os.system("python ml_adaptive_v49_alerts_fixed.py --capital 16000")

# Step 3: Run test
print("\n3. Running test scan...")
print("-"*60)
os.system("python ml_adaptive_v49_alerts_fixed.py --once")

print("\n" + "="*60)
print("SETUP COMPLETE!")
print("="*60)
print("\nIMPORTANT: Gmail Setup Required")
print("-"*60)
print("You got an authentication error because you need an")
print("App-Specific Password (not your regular Gmail password)")
print("")
print("To fix:")
print("1. Go to: https://myaccount.google.com/apppasswords")
print("2. Sign in to your Google account")
print("3. Generate a password for 'Mail'")
print("4. Use that 16-character password")
print("")
print("Then run:")
print("python ml_adaptive_v49_alerts_fixed.py \\")
print("  --email lestinjackson@gmail.com \\")
print("  --password YOUR_16_CHAR_APP_PASSWORD")
print("")
print("Or run WITHOUT email for now:")
print("python ml_adaptive_v49_alerts_fixed.py")
print("="*60)
