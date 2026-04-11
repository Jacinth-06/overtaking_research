import smbus, time

bus = smbus.SMBus(1)
addr = 0x40

def write_reg(reg, val):
    bus.write_byte_data(addr, reg, val)

def read_reg(reg):
    return bus.read_byte_data(addr, reg)

def pca_init():
    pre = round(25_000_000 / (4096 * 50)) - 1
    old = read_reg(0x00)
    write_reg(0x00, (old & 0x7F) | 0x10)
    write_reg(0xFE, pre)
    write_reg(0x00, old)
    time.sleep(0.005)
    write_reg(0x00, old | 0xA0)

def set_us(ch, us):
    count = int(us / 20000 * 4096)
    base = 0x06 + ch * 4
    bus.write_i2c_block_data(addr, base, [0, 0, count & 0xFF, count >> 8])

pca_init()
print("PCA9685 ready at 50Hz")

print("Sending neutral to all channels...")
for ch in range(16):
    set_us(ch, 1500)

time.sleep(2)
print("Done — ESC should be armed and stopped")
bus.close()