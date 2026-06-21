#!/usr/bin/env python3
"""
lidar_probe.py — print raw lidar scan to find cone angles
Place an object in different positions and watch which angles light up.
Run: python lidar_probe.py
"""

import time
from jetracer import JetRacer

car = JetRacer(init_lidar=True)
car.arm(delay=3)

print("Scanning... place object in front-right and watch angles drop\n")

while True:
    scan = car.lidar_scan(samples=150)
    
    # Sort by angle
    sorted_scan = sorted(scan.items())
    
    # Only print angles with a valid reading (filter noise)
    hits = [(ang, dist) for ang, dist in sorted_scan if dist > 10]
    
    print("\033[2J\033[H")  # clear terminal
    print(f"{'Angle':>6}  {'Dist(mm)':>10}  {'Bar'}")
    print("-" * 40)
    
    for ang, dist in hits:
        bar = "█" * min(int(dist / 50), 40)  # visual bar, capped at 2000mm
        marker = " <-- HERE" if dist < 400 else ""  # highlight close objects
        print(f"{ang:>6}°  {dist:>8.0f}mm  {bar}{marker}")
    
    print(f"\nTotal readings: {len(hits)}")
    time.sleep(0.2)