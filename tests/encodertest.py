import serial
import struct

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
print("Drive the car forward slowly. Watching bytes 18-19...\n")
print(f"{'Count':>8} {'Delta':>8} {'Hex18-19':>10}")
print("-" * 35)

last_count = None
packet_count = 0

while True:
    try:
        if ser.read(1) == b'\x55':
            data = ser.read(26)
            if len(data) == 26:
                # Show multiple interpretations of bytes 18-19 and nearby
                b18_19_be = struct.unpack('>H', data[18:20])[0]   # big-endian uint16
                b18_19_le = struct.unpack('<H', data[18:20])[0]   # little-endian uint16
                b17_18_be = struct.unpack('>H', data[17:19])[0]
                b19_20_be = struct.unpack('>H', data[19:21])[0]

                delta = b18_19_be - last_count if last_count is not None else 0
                last_count = b18_19_be

                # Only print every 10th packet to reduce noise
                packet_count += 1
                if packet_count % 10 == 0:
                    print(f"BE[18:20]={b18_19_be:6d}  LE[18:20]={b18_19_le:6d}  "
                          f"BE[17:19]={b17_18_be:6d}  BE[19:21]={b19_20_be:6d}  "
                          f"delta={delta:+5d}  hex={data[18:21].hex()}")

    except KeyboardInterrupt:
        print("\nDone.")
        ser.close()
        break
    except Exception as e:
        print(f"err: {e}")