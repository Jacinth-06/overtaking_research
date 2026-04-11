from jetracer import JetRacer
import time

print("=" * 40)
print("   JetRacer System Test")
print("=" * 40)

print("\n[1/7] Initializing JetRacer...")
try:
    car = JetRacer(init_lidar=True)
    print("Init: PASS ✓")
except Exception as e:
    print(f"Init: FAIL ✗ — {e}")
    exit()

print("\n[2/7] Arming ESC...")
try:
    car.arm(delay=3)
    print("Arm: PASS ✓")
except Exception as e:
    print(f"Arm: FAIL ✗ — {e}")

print("\n[3/7] Testing steering...")
try:
    print("  Center...")
    car.steer_center()
    time.sleep(1)
    print("  Left 50%...")
    car.steer_left(0.5)
    time.sleep(1)
    print("  Right 50%...")
    car.steer_right(0.5)
    time.sleep(1)
    print("  Center...")
    car.steer_center()
    time.sleep(1)
    print("Steering: PASS ✓")
except Exception as e:
    print(f"Steering: FAIL ✗ — {e}")

print("\n[4/7] Testing throttle — LIFT CAR OFF GROUND!")
input("Press Enter when car is lifted...")
try:
    print("  Slow forward 15%...")
    car.forward(0.15)
    time.sleep(2)
    print("  Stop...")
    car.stop()
    time.sleep(1)
    print("Throttle: PASS ✓")
except Exception as e:
    print(f"Throttle: FAIL ✗ — {e}")
    car.stop()

print("\n[5/7] Checking lidar health...")
try:
    result = car.lidar_health()
    if result == "GOOD":
        print("Lidar health: PASS ✓")
    else:
        print(f"Lidar health: WARNING — status={result}")
except Exception as e:
    print(f"Lidar health: FAIL ✗ — {e}")

print("\n[6/7] Running lidar scan...")
try:
    car.lidar_summary()
    print("Lidar scan: PASS ✓")
except Exception as e:
    print(f"Lidar scan: FAIL ✗ — {e}")

print("\n[7/7] Detecting closest object...")
try:
    angle, dist = car.lidar_closest()
    if dist:
        if dist < 300:
            print(f"WARNING — object very close at {dist:.1f}mm!")
        else:
            print(f"Closest object: {dist:.1f}mm at {angle}°")
        print("Closest detect: PASS ✓")
    else:
        print("Closest detect: no reading")
except Exception as e:
    print(f"Closest detect: FAIL ✗ — {e}")

print("\n" + "=" * 40)
print("   Test Complete")
print("=" * 40)

car.close()