import serial
import struct
import time
from jetracer import JetRacer

# ── Init ──────────────────────────────────────────────────────────────────────
car = JetRacer(init_lidar=False)
car.arm(delay=2)

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
print("Starting in 2 seconds — will drive forward for 3s, then stop.\n")
print(f"{'Phase':<12} {'BE[18:20]':>10} {'LE[18:20]':>10} {'BE[17:19]':>10} {'BE[19:21]':>10} {'delta':>7}  hex")
print("-" * 80)

time.sleep(2)

# ── Phase control ─────────────────────────────────────────────────────────────
start      = time.time()
phase      = "STATIONARY1"
drove      = False
stopped    = False

last_count   = None
packet_count = 0

while True:
    try:
        now = time.time() - start

        # Phase transitions
        if now >= 3.0 and phase == "STATIONARY1":
            phase = "DRIVING"
            car.forward(0.25)
            print("\n>>> DRIVING FORWARD <<<\n")

        if now >= 6.0 and phase == "DRIVING":
            phase = "STATIONARY2"
            car.stop()
            print("\n>>> STOPPED <<<\n")

        if now >= 9.0:
            print("\nDone.")
            car.stop()
            ser.close()
            break

        # ── Read serial ───────────────────────────────────────────────────────
        if ser.read(1) == b'\x55':
            data = ser.read(26)
            if len(data) == 26:
                b18_19_be = struct.unpack('>H', data[18:20])[0]
                b18_19_le = struct.unpack('<H', data[18:20])[0]
                b17_18_be = struct.unpack('>H', data[17:19])[0]
                b19_20_be = struct.unpack('>H', data[19:21])[0]

                delta = b18_19_be - last_count if last_count is not None else 0
                last_count = b18_19_be

                packet_count += 1
                if packet_count % 10 == 0:
                    print(f"{phase:<12} {b18_19_be:>10}  {b18_19_le:>10}  "
                          f"{b17_18_be:>10}  {b19_20_be:>10}  {delta:>+7}  "
                          f"{data[17:21].hex()}")

    except KeyboardInterrupt:
        car.stop()
        ser.close()
        print("\nInterrupted.")
        break
    except Exception as e:
        print(f"err: {e}")