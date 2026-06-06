import serial
import struct

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
print("Packet structure analysis...\n")

packets = []
while len(packets) < 20:
    if ser.read(1) == b'\x55':
        data = ser.read(30)
        if len(data) == 30:
import serial

ser = serial.Serial('/dev/ttyACM1', 115200, timeout=2)
print("Raw bytes from ttyACM1 (hex):")
for _ in range(10):
    raw = ser.read(30)
    if raw:
        print(' '.join(f'{b:02X}' for b in raw))
ser.close()