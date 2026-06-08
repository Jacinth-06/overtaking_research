#!/usr/bin/env python3
"""
version3.py — Phase 1: Build State Estimation
Tracks vehicle position (x, y, yaw, speed) at all times using:
- Wheel Encoders (for speed and distance travelled)
- IMU Gyro Z (for heading / yaw)
- Dead Reckoning kinematic updates

Reads from Waveshare RP2040 board over /dev/ttyACM0.
"""

import sys
import time
import math
import serial
from dataclasses import dataclass

# SPEED_SCALE maps raw RP2040 encoder readings to m/s.
# Empirically calibrated: 1.12m actual distance / 10.8m raw reported (0.1037 multiplier on average)
# Here we reuse the scale from version2.py (0.00748).
SPEED_SCALE = 0.00748

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


def main():
    # Attempt to open the serial port
    serial_port = '/dev/ttyACM0'
    baud_rate = 115200
    print(f"[init] Connecting to serial port: {serial_port} at {baud_rate} baud...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1.0)
    except Exception as e:
        print(f"\n[error] Failed to open serial port {serial_port}: {e}")
        print("Please check if the rover is connected and `/dev/ttyACM0` exists.")
        sys.exit(1)

    # 1. Perform gyro calibration
    gz_bias = calibrate_gyro(ser, duration=2.0)

    # Initialize state variables
    state = State(x=0.0, y=0.0, yaw=0.0, speed=0.0)
    
    last_time = time.time()
    packet_count = 0
    fps_time = time.time()
    freq = 0.0

    print("\n" + "="*50)
    print("State Estimation Active (Phase 1)")
    print("Push the rover forward/backward and turn to verify.")
    print("Press Ctrl+C to stop.")
    print("="*50 + "\n")

    try:
        while True:
            parsed = parse_telemetry_packet(ser)
            if parsed is None:
                continue

            # Calculate dt
            now = time.time()
            dt = now - last_time
            last_time = now

            gz_deg, lvel, rvel = parsed

            # Measure frequency
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
            # Format outputs:
            # - x, y in meters
            # - yaw in degrees for easy verification
            # - speed in meters/second
            # - loop frequency in Hz
            yaw_deg = math.degrees(state.yaw)
            print(
                f"\rState -> X: {state.x:6.3f} m | Y: {state.y:6.3f} m | "
                f"Yaw: {yaw_deg:6.1f}° | Speed: {state.speed:5.3f} m/s | "
                f"Freq: {freq:4.1f} Hz",
                end="",
                flush=True
            )

    except KeyboardInterrupt:
        print("\n\n[exit] Exiting state estimation.")
    finally:
        ser.close()
        print("[exit] Serial port closed.")


if __name__ == '__main__':
    main()
