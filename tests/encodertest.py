import serial
import struct
import time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
print("Watching for both IMU and encoder packets...\n")

while True:
    try:
        if ser.read(1) == b'\x55':
            # Try 12 bytes (IMU)
            data = ser.read(12)
            
            # Check if it looks like encoder (8 bytes meaningful, last 4 zeros)
            # by trying both interpretations
            
            # Try as encoder (first 8 bytes)
            if len(data) >= 8:
                left, right = struct.unpack('>ii', data[0:8])
                ax, ay, az, gx, gy, gz = struct.unpack('>hhhhhh', data[0:12])
                
                # Encoder values should be small integers that grow slowly
                # IMU values should be small after dividing by 16384/131
                enc_plausible = abs(left) < 500000 and abs(right) < 500000
                imu_plausible = abs(ax/16384.0) < 3 and abs(gz/131.0) < 500
                
                print(f"RAW 12 bytes: {data.hex()}")
                print(f"  As encoder: left={left}, right={right}  {'← ENCODER?' if enc_plausible else ''}")
                print(f"  As IMU:     ax={ax/16384:.2f}g, gz={gz/131:.1f}°/s  {'← IMU?' if imu_plausible else ''}")
                print()
                
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"err: {e}")

ser.close()