from jetracer import JetRacer
import time

# Initialize car and Lidar
car = JetRacer(init_lidar=True)
car.arm(delay=3)

# Configuration
STOP_DISTANCE = 800.0  # Stop if object is closer than 800mm (80cm)
DRIVE_SPEED = 0.4     # Slow crawl for testing

print(f"Safety Shield Active. Threshold: {STOP_DISTANCE}mm")
print("Car will move forward in 2 seconds... LIFT CAR IF UNSURE!")
time.sleep(2)

try:
    while True:
        # 1. Get a quick scan
        scan = car.lidar_scan(samples=120) # Using fewer samples for faster reaction
        
        # 2. Extract the minimum distance in the front 40-degree cone
        # Angles: 340-359 and 0-20
        front_distances = [dist for ang, dist in scan.items() if ang >= 340 or ang <= 20]
        
        if front_distances:
            closest_front = min(front_distances)
            print(f"Distance Ahead: {closest_front:.1f}mm", end='\r')
            
            # 3. Decision Logic
            if closest_front < STOP_DISTANCE:
                car.stop()
                print(f"\n[!] EMERGENCY STOP: Object at {closest_front:.1f}mm")
                # Wait until path is clear or script is exited
                while closest_front < STOP_DISTANCE:
                    time.sleep(0.5)
                    # Re-check
                    new_scan = car.lidar_scan(samples=100)
                    new_front = [d for a, d in new_scan.items() if a >= 340 or a <= 20]
                    closest_front = min(new_front) if new_front else 0
                print("\n[+] Path clear. Resuming...")
            else:
                car.forward(DRIVE_SPEED)
        else:
            # If Lidar misses a frame, stop for safety
            car.stop()
            
except KeyboardInterrupt:
    print("\nUser stopped script.")
finally:
    car.close()