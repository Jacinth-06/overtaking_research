#!/usr/bin/env python3
"""
version3.py — Phase 1: Build State Estimation (Automated Drive Verification + Lidar Trigger)
Tracks vehicle position (x, y, yaw, speed) at all times using:
- Wheel Encoders (for speed and distance travelled)
- IMU Gyro Z (for heading / yaw)
- Dead Reckoning kinematic updates

Performs an automated test sequence:
1. Calibrate gyro (2 seconds stationary)
2. Drive straight forward for 0.5 meters, then stop
3. Turn right by 90 degrees, then stop
If a front obstacle is detected within 800mm, it immediately triggers the OVERTAKE state.
"""

import os
import sys
import time
import math
import serial
import types
import threading
from dataclasses import dataclass

# Add parent directory to path so we can import local jetracer package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jetracer import JetRacer

# SPEED_SCALE maps raw RP2040 encoder readings to m/s.
SPEED_SCALE = 0.00748

# Target parameters for automated verification
FORWARD_TARGET_DIST = 0.5   # Target forward distance in meters
TURN_TARGET_DEG = 90.0      # Target turn angle in degrees

# Lidar obstacle detection cache
_lidar_cache = {"closest": 0.0, "blocked": False}
_lidar_cache_lock = threading.Lock()

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0      # In radians
    speed: float = 0.0    # In m/s


def lidar_loop(car: JetRacer):
    """
    Background thread to poll Lidar and cache front distances.
    Detects obstacles at 800mm distance.
    """
    print("[lidar] Background safety thread started")
    while True:
        try:
            # Scan front cone (320-360 deg and 0-40 deg)
            scan = car.lidar_scan(samples=150)
            front_distances = [dist for ang, dist in scan.items() if (ang >= 320 or ang <= 40) and dist > 10]
            closest_front = min(front_distances) if front_distances else 0.0
            
            # Block/trigger if obstacle is closer than 800mm
            is_blocked = 0.0 < closest_front < 800.0
            
            with _lidar_cache_lock:
                _lidar_cache["closest"] = round(closest_front, 1)
                _lidar_cache["blocked"] = is_blocked
        except Exception as e:
            # Do not crash the thread on scan error
            pass
        time.sleep(0.10)


def parse_telemetry_packet(ser, head1=0xAA, head2=0x55):
    """
    Parses a single protocol packet from the serial connection.
    Packet format: AA 55 2D 01 [39 data bytes] [checksum]
    Total frame size = 45 bytes.
    """
    b = ser.read(1)
    if not b or b[0] != head1:
        return None

    b = ser.read(1)
    if not b or b[0] != head2:
        return None

    b = ser.read(1)
    if not b:
        return None
    frame_size = b[0]

    if frame_size < 5 or frame_size > 50:
        return None

    remaining = frame_size - 3
    rest = ser.read(remaining)
    if len(rest) != remaining:
        return None

    frame = bytes([head1, head2, frame_size]) + rest

    calc_sum = sum(frame[:-1]) & 0xFF
    recv_sum = frame[-1]
    if calc_sum != recv_sum:
        return None

    gz_raw = int.from_bytes(frame[8:10], 'big', signed=True)
    gz_deg = (gz_raw / 32768.0) * 2000.0

    lvel = int.from_bytes(frame[34:36], 'big', signed=True)
    rvel = int.from_bytes(frame[36:38], 'big', signed=True)

    return gz_deg, lvel, rvel


def calibrate_gyro(ser, duration=2.0):
    """
    Measures and returns the average Gyro Z bias (drift offset) over the specified duration.
    Rover must remain stationary during calibration.
    """
    print(f"\n[init] Keep rover completely still. Calibrating gyroscope bias for {duration}s...")
    start_time = time.time()
    samples = []
    ser.reset_input_buffer()

    while time.time() - start_time < duration:
        parsed = parse_telemetry_packet(ser)
        if parsed is not None:
            gz_deg, _, _ = parsed
            samples.append(gz_deg)
        time.sleep(0.01)

    if not samples:
        print("[warning] No calibration samples collected. Defaulting bias to 0.0")
        return 0.0

    bias = sum(samples) / len(samples)
    print(f"[init] Calibration successful over {len(samples)} samples.")
    print(f"[init] Gyro Z average bias offset: {bias:.4f} deg/s")
    return bias


# --- Silent overrides for JetRacer driver to keep output clean ---
def silent_steer(self, val):
    val = max(-1.0, min(1.0, val))
    if val >= 0:
        us = self.STEER_CENTER + val * (self.STEER_RIGHT - self.STEER_CENTER)
    else:
        us = self.STEER_CENTER + val * (self.STEER_CENTER - self.STEER_LEFT)
    self._set_us(self.STEER_CH, us)

def silent_throttle(self, val):
    val = max(-1.0, min(1.0, val))
    if val >= 0:
        us = self.THROTTLE_NEUTRAL + val * (self.THROTTLE_FWD_MAX - self.THROTTLE_NEUTRAL)
    else:
        us = self.THROTTLE_NEUTRAL + val * (self.THROTTLE_NEUTRAL - self.THROTTLE_REV_MAX)
    self._set_us(self.THROTTLE_CH, us)

def silent_stop(self):
    self._set_us(self.THROTTLE_CH, self.THROTTLE_NEUTRAL)
    self._set_us(self.STEER_CH, self.STEER_CENTER)


def main():
    # 1. Initialize JetRacer (PCA9685/Servo/Motor + RPLidar A1)
    print("[init] Initializing JetRacer motor and Lidar controllers...")
    try:
        # Enable init_lidar to match the requirement
        car = JetRacer(init_lidar=True)
        car.arm(delay=2)
        
        # Monkey-patch driver to prevent console logging on actuator updates
        car.steer = types.MethodType(silent_steer, car)
        car.throttle = types.MethodType(silent_throttle, car)
        car.stop = types.MethodType(silent_stop, car)
        
    except Exception as e:
        print(f"\n[error] Failed to initialize JetRacer: {e}")
        sys.exit(1)

    # 2. Attempt to open the serial port for RP2040 communication
    serial_port = '/dev/ttyACM0'
    baud_rate = 115200
    print(f"[init] Connecting to serial port: {serial_port} at {baud_rate} baud...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1.0)
    except Exception as e:
        print(f"\n[error] Failed to open serial port {serial_port}: {e}")
        car.stop()
        sys.exit(1)

    # 3. Perform gyro calibration
    gz_bias = calibrate_gyro(ser, duration=2.0)

    # 4. Start the background Lidar thread
    lt_thread = threading.Thread(target=lidar_loop, args=(car,), daemon=True)
    lt_thread.start()

    # Initialize state variables
    state = State(x=0.0, y=0.0, yaw=0.0, speed=0.0)
    
    last_time = time.time()
    packet_count = 0
    fps_time = time.time()
    freq = 0.0

    # Test sequence phases
    PHASE_DRIVE_FORWARD = 1
    PHASE_PAUSE = 2
    PHASE_TURN_RIGHT = 3
    PHASE_DONE = 4
    PHASE_OVERTAKE = 5
    
    current_phase = PHASE_DRIVE_FORWARD
    phase_start_time = time.time()
    phase_start_yaw = 0.0

    print("\n" + "="*60)
    print("Automated State Estimation & Lidar Trigger Verification")
    print(f"  Step 1: Drive forward {FORWARD_TARGET_DIST} meters")
    print("  Trigger: Obstacle < 800mm switches to OVERTAKE state")
    print("="*60 + "\n")

    try:
        while current_phase != PHASE_DONE:
            parsed = parse_telemetry_packet(ser)
            if parsed is None:
                continue

            # Calculate dt
            now = time.time()
            dt = now - last_time
            last_time = now

            gz_deg, lvel, rvel = parsed

            # Measure loop frequency
            packet_count += 1
            if now - fps_time >= 1.0:
                freq = packet_count / (now - fps_time)
                packet_count = 0
                fps_time = now

            if dt <= 0:
                continue

            # ── 1. Calculate Gyro Heading (Yaw) ──
            gz_calibrated = gz_deg - gz_bias
            omega = math.radians(gz_calibrated)
            state.yaw += omega * dt
            state.yaw = (state.yaw + math.pi) % (2.0 * math.pi) - math.pi

            # ── 2. Calculate Encoder Speed ──
            avg_vel = (lvel + rvel) / 2.0
            state.speed = avg_vel * SPEED_SCALE

            # ── 3. Update Dead Reckoning Position ──
            state.x += state.speed * math.cos(state.yaw) * dt
            state.y += state.speed * math.sin(state.yaw) * dt

            # ── 4. Check Lidar Trigger ──
            with _lidar_cache_lock:
                lidar_blocked = _lidar_cache["blocked"]
                closest_dist = _lidar_cache["closest"]

            if lidar_blocked and current_phase != PHASE_OVERTAKE:
                current_phase = PHASE_OVERTAKE
                print(f"\n[LIDAR TRIGGER] Obstacle detected at {closest_dist} mm! Switched to OVERTAKE state.")

            # ── 5. Automated Phase Logic ──
            if current_phase == PHASE_DRIVE_FORWARD:
                car.steer(0.0)
                car.throttle(0.15)
                
                distance_travelled = math.sqrt(state.x**2 + state.y**2)
                if distance_travelled >= FORWARD_TARGET_DIST:
                    car.stop()
                    current_phase = PHASE_PAUSE
                    phase_start_time = time.time()
                    print(f"\n[auto] Forward phase complete. Reached {distance_travelled:.3f}m. Pausing...")

            elif current_phase == PHASE_PAUSE:
                car.stop()
                if time.time() - phase_start_time >= 1.0:
                    current_phase = PHASE_TURN_RIGHT
                    phase_start_yaw = state.yaw
                    print("[auto] Pause complete. Starting turn phase...")

            elif current_phase == PHASE_TURN_RIGHT:
                car.steer(0.8)
                car.throttle(0.15)
                
                yaw_change = math.degrees(abs(state.yaw - phase_start_yaw))
                if yaw_change > 180.0:
                    yaw_change = 360.0 - yaw_change
                    
                if yaw_change >= TURN_TARGET_DEG:
                    car.stop()
                    current_phase = PHASE_DONE
                    print(f"\n[auto] Turn phase complete. Turned {yaw_change:.1f}°. Verification finished successfully!")

            elif current_phase == PHASE_OVERTAKE:
                # Overtake triggered: stop motors (only trigger that's it)
                car.stop()

            # ── 6. Live Terminal Display ──
            yaw_deg = math.degrees(state.yaw)
            phase_labels = {
                PHASE_DRIVE_FORWARD: "DRIVING FORWARD",
                PHASE_PAUSE: "PAUSED",
                PHASE_TURN_RIGHT: "TURNING RIGHT",
                PHASE_OVERTAKE: "OVERTAKE STATE TRIGGERED"
            }
            print(
                f"\r[{phase_labels.get(current_phase, 'UNKNOWN')}] "
                f"X: {state.x:6.3f} m | Y: {state.y:6.3f} m | "
                f"Yaw: {yaw_deg:6.1f}° | Speed: {state.speed:5.3f} m/s | "
                f"Freq: {freq:4.1f} Hz\033[K",
                end="",
                flush=True
            )

    except KeyboardInterrupt:
        print("\n\n[exit] KeyboardInterrupt detected. Stopping car...")
    finally:
        car.stop()
        ser.close()
        print(f"\nFinal State -> X: {state.x:.3f} m | Y: {state.y:.3f} m | Yaw: {math.degrees(state.yaw):.1f}°")
        print("[exit] Serial port closed.")


if __name__ == '__main__':
    main()
