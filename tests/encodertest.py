import smbus2
import time
import threading
from jetracer import JetRacer

# ── Init ──────────────────────────────────────────────────────────────────────
car = JetRacer(init_lidar=False)
car.arm(delay=2)

bus  = smbus2.SMBus(1)
ADDR = 0x60

# ── Snapshot registers at rest ────────────────────────────────────────────────
def read_all_regs():
    snapshot = {}
    for reg in range(0x00, 0x40):
        try:
            data = bus.read_i2c_block_data(ADDR, reg, 4)
            snapshot[reg] = data
        except Exception:
            pass
    return snapshot

print("Reading registers at REST...")
rest = read_all_regs()
time.sleep(0.5)

# ── Drive forward 3 seconds ───────────────────────────────────────────────────
print("Driving forward 3s...")
car.forward(0.25)
time.sleep(3)
car.stop()
print("Stopped.\n")
time.sleep(0.3)

# ── Snapshot registers after driving ─────────────────────────────────────────
print("Reading registers AFTER DRIVE...")
after = read_all_regs()

# ── Compare: show only registers that CHANGED ─────────────────────────────────
print(f"\n{'Reg':>5}  {'REST bytes':>12}  {'AFTER bytes':>12}  "
      f"{'REST i32':>12}  {'AFTER i32':>12}  {'DELTA':>10}")
print("-" * 75)

changed = False
for reg in sorted(rest.keys()):
    if reg not in after:
        continue
    r = rest[reg]
    a = after[reg]
    if r != a:
        r_i32 = int.from_bytes(r, 'big', signed=True)
        a_i32 = int.from_bytes(a, 'big', signed=True)
        delta  = a_i32 - r_i32
        print(f"  {reg:#04x}  {r.hex():>12}  {a.hex():>12}  "
              f"{r_i32:>12}  {a_i32:>12}  {delta:>+10}")
        changed = True

if not changed:
    print("  No registers changed — encoder not exposed via I2C at 0x60")
    print("\n  Printing ALL readable registers for reference:")
    print(f"\n  {'Reg':>5}  {'Bytes':>12}  {'i16 BE':>10}  {'i32 BE':>12}")
    print("  " + "-" * 50)
    for reg, data in sorted(rest.items()):
        i16 = int.from_bytes(data[0:2], 'big', signed=True)
        i32 = int.from_bytes(data[0:4], 'big', signed=True)
        print(f"    {reg:#04x}  {data.hex():>12}  {i16:>10}  {i32:>12}")

bus.close()
car.stop()