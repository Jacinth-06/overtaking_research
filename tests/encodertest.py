import serial
import struct

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
print("Packet structure analysis...\n")

packets = []
while len(packets) < 20:
    if ser.read(1) == b'\x55':
        data = ser.read(30)
        if len(data) == 30:
            packets.append(data)

# Show which bytes actually change across packets (those carry real data)
print("Byte | Values seen across 20 packets")
print("-----|" + "-"*60)
for i in range(30):
    vals = [p[i] for p in packets]
    unique = set(vals)
    changes = len(unique) > 1
    marker = " ← LIVE DATA" if changes else "   (static)"
    print(f"  {i:02d} | {vals[:8]}  {marker}")

ser.close()