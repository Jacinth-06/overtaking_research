import serial
import struct
import time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
print("Listening for 0x55 start byte... (Ctrl+C to stop)\n")

count = 0
while True:
    try:
        byte = ser.read(1)
        if byte == b'\x55':
            # Read next 30 bytes after start byte — enough to see full packet
            rest = ser.read(30)
            hex_str = ' '.join(f'{b:02X}' for b in rest)
            print(f"[{count:04d}] 55 | {hex_str}")
            count += 1

            # Try interpreting bytes at different offsets as int32 pairs
            for offset in range(0, min(len(rest)-7, 20)):
                try:
                    l, r = struct.unpack('>ii', rest[offset:offset+8])
                    if abs(l) < 1_000_000 and abs(r) < 1_000_000:
                        print(f"        offset[{offset}]: left={l}, right={r}  ← plausible encoder values?")
                except:
                    pass

    except KeyboardInterrupt:
        print("\nDone.")
        ser.close()
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(0.1)