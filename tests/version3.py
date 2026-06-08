#!/usr/bin/env python3
"""
version3.py — Phase 1: Build State Estimation
Tracks vehicle position (x, y, yaw, speed) at all times using:
- Wheel Encoders (for speed and distance travelled)
- IMU Gyro Z (for heading / yaw)
- Dead Reckoning kinematic updates

Reads from Waveshare RP2040 board over /dev/ttyACM0.
Includes keyboard WASD drive control thread for manual operation.
"""

import os
import sys
import time
import math
import select
import termios
import tty
import serial
import threading
from dataclasses import dataclass

# Add parent directory to path so we can import local jetracer package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jetracer import JetRacer

# SPEED_SCALE maps raw RP2040 encoder readings to m/s.
SPEED_SCALE = 0.00748

# Global control flags for keyboard thread communication
stop_keyboard = False
current_throttle = 0.0
current_steer = 0.0

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0      # In radians
    speed: float = 0.0    # In m/s


def parse_telemetry_packet(ser, head1=0xAA, head2=0x55):
    """
    Parses a single protocol packet from the serial connection.
    Packet format: AA 55 2D 01 [39 data bytes] [checksum]
    Total frame size = 45 bytes.
    """
    # 1. Wait for HEAD1
    b = ser.read(1)
    if not b or b[0] != head1:
        return None

    # 2. Wait for HEAD2
    b = ser.read(1)
    if not b or b[0] != head2:
        return None

    # 3. Read size byte
    b = ser.read(1)
    if not b:
        return None
    frame_size = b[0]

    # Validate size bounds
    if frame_size < 5 or frame_size > 50:
        return None

    # 4. Read remaining bytes
    remaining = frame_size - 3
    rest = ser.read(remaining)
    if len(rest) != remaining:
        return None

    # 5. Construct full frame
    frame = bytes([head1, head2, frame_size]) + rest

    # 6. Verify checksum
    calc_sum = sum(frame[:-1]) & 0xFF
    recv_sum = frame[-1]
    if calc_sum != recv_sum:
        return None

    # 7. Extract Gyro Z and encoder velocities
    # Gyro Z is in frame[8:10] (16-bit signed big-endian)
    gz_raw = int.from_bytes(frame[8:10], 'big', signed=True)
    gz_deg = (gz_raw / 32768.0) * 2000.0  # Convert to degrees/sec

    # Encoder velocities are in frame[34:36] and frame[36:38]
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

    # Clear serial buffer to ensure fresh readings
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


def keyboard_control_thread(car, old_settings):
    """
    Background thread that reads keyboard presses and updates car actuators.
    Controls:
      W: Forward (0.15 throttle)
      S: Reverse (-0.15 throttle)
      A: Steer Left (-0.60 steering)
      D: Steer Right (0.60 steering)
      Space / X: Stop throttle and center steering
      Q: Quit application cleanly
    """
    global stop_keyboard, current_throttle, current_steer
    
    print("\n" + "="*50)
    print("Keyboard Drive Controls Active:")
    print("  W: Forward | S: Reverse")
    print("  A: Left    | D: Right")
    print("  Space / X: Stop & Center")
    print("  Q: Quit")
    print("="*50 + "\n")

    try:
        while not stop_keyboard:
            # Configure stdin to raw non-blocking character reading
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            
            if rlist:
                key = sys.stdin.read(1)
                
                # Temporarily restore standard terminal settings to prevent print corruption
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
                k = key.lower()
                if k == 'q':
                    print("\n[drive] Shutting down application...")
                    stop_keyboard = True
                    break
                elif k == 'w':
                    current_throttle = 0.15
                    car.throttle(current_throttle)
                elif k == 's':
                    current_throttle = -0.15
                    car.throttle(current_throttle)
                elif k == 'a':
                    current_steer = -0.6
                    car.steer(current_steer)
                elif k == 'd':
                    current_steer = 0.6
                    car.steer(current_steer)
                elif key == ' ' or k == 'x':
                    current_throttle = 0.0
                    current_steer = 0.0
                    car.stop()
            else:
                # Keep terminal settings standard while idle to allow prints
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
    except Exception as e:
        print(f"\n[drive] Keyboard thread error: {e}")
    finally:
        # Final safety restore of terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    global stop_keyboard, current_throttle, current_steer

    # Save original terminal settings for restoration later
    old_terminal_settings = termios.tcgetattr(sys.stdin)

    # 1. Initialize JetRacer (PCA9685/Servo/Motor)
    # We do not initialize Lidar for Phase 1 to keep startup fast and independent
    print("[init] Initializing JetRacer motor controller...")
    try:
        car = JetRacer(init_lidar=False)
        car.arm(delay=2)
    except Exception as e:
        print(f"\n[error] Failed to initialize JetRacer motor board: {e}")
        print("Please verify connection to PCA9685 I2C interface.")
        sys.exit(1)

    # 2. Attempt to open the serial port for RP2040 communication
    serial_port = '/dev/ttyACM0'
    baud_rate = 115200
    print(f"[init] Connecting to serial port: {serial_port} at {baud_rate} baud...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1.0)
    except Exception as e:
        print(f"\n[error] Failed to open serial port {serial_port}: {e}")
        print("Please check if the rover is connected and `/dev/ttyACM0` exists.")
        car.stop()
        sys.exit(1)

    # 3. Perform gyro calibration
    gz_bias = calibrate_gyro(ser, duration=2.0)

    # 4. Start the background keyboard control thread
    kb_thread = threading.Thread(
        target=keyboard_control_thread,
        args=(car, old_terminal_settings),
        daemon=True
    )
    kb_thread.start()

    # Initialize state variables
    state = State(x=0.0, y=0.0, yaw=0.0, speed=0.0)
    
    last_time = time.time()
    packet_count = 0
    fps_time = time.time()
    freq = 0.0

    print("\nState Estimation Active (Phase 1)")
    print("="*50 + "\n")

    try:
        while not stop_keyboard:
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
            # Subtract bias and convert to radians/sec
            gz_calibrated = gz_deg - gz_bias
            omega = math.radians(gz_calibrated)
            
            # Integrate yaw
            state.yaw += omega * dt
            
            # Normalize yaw to [-pi, pi]
            state.yaw = (state.yaw + math.pi) % (2.0 * math.pi) - math.pi

            # ── 2. Calculate Encoder Speed ──
            avg_vel = (lvel + rvel) / 2.0
            state.speed = avg_vel * SPEED_SCALE

            # ── 3. Update Dead Reckoning Position ──
            state.x += state.speed * math.cos(state.yaw) * dt
            state.y += state.speed * math.sin(state.yaw) * dt

            # ── 4. Live Terminal Display ──
            yaw_deg = math.degrees(state.yaw)
            print(
                f"\rState -> X: {state.x:6.3f} m | Y: {state.y:6.3f} m | "
                f"Yaw: {yaw_deg:6.1f}° | Speed: {state.speed:5.3f} m/s | "
                f"Freq: {freq:4.1f} Hz | "
                f"Drive -> T: {current_throttle:+.2f}, S: {current_steer:+.2f}",
                end="",
                flush=True
            )

    except KeyboardInterrupt:
        print("\n\n[exit] KeyboardInterrupt detected.")
    finally:
        # Stop car motors safely
        car.stop()
        # Close serial port
        ser.close()
        # Ensure terminal settings are restored to default
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)
        print("[exit] Cleared motor commands, closed serial port, and restored terminal settings.")


if __name__ == '__main__':
    main()
