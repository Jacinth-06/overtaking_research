import serial
import struct

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
print("Spin the wheels NOW. Watching for ANY value that changes with wheel movement...\n")

prev_bytes = None
count = 0

while True:
    try:
        if ser.read(1) == b'\x55':
            # Read 30 bytes to catch any packet size
            data = ser.read(30)
            hex_str = data.hex()
            
            # Show bytes 8-16 specifically (where encoder would be if after IMU)
            if len(data) >= 16:
                # Try little-endian too (RP2040 is little-endian natively)
                l_le, r_le = struct.unpack('<ii', data[0:8])
                l_le2, r_le2 = struct.unpack('<ii', data[8:16])
                
                print(f"[{count:04d}] full hex: {hex_str}")
                print(f"  bytes[0:8]  little-endian int32 pair: {l_le}, {r_le}")
                print(f"  bytes[8:16] little-endian int32 pair: {l_le2}, {r_le2}")
                print()
                count += 1
                
    except KeyboardInterrupt:
        break

ser.close()