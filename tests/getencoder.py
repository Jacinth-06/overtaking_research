import serial
import struct

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

def parse_encoder():
    while True:
        if ser.read() == b'\x55':
            data = ser.read(8)  # Adjust size based on your packet format
            
            if len(data) == 8:
                try:
                    # Option A: Two 32-bit signed ints (encoder left & right)
                    left, right = struct.unpack('>ii', data)
                    print(f"Encoder Left: {left} | Right: {right}")

                    # Option B: Single 32-bit int + checksum (adjust as needed)
                    # count, checksum = struct.unpack('>iI', data)
                    # print(f"Encoder Count: {count}")

                except struct.error:
                    pass

try:
    print("Reading encoder stream...")
    parse_encoder()
except KeyboardInterrupt:
    ser.close()