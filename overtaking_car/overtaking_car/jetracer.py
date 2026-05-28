import smbus, serial, time

class JetRacer:
    """
    Waveshare JetRacer unified controller
    Handles PCA9685 (steering + throttle) and RPLidar A1
    """

    # --- PCA9685 config ---
    PCA9685_ADDR   = 0x40
    PCA9685_BUS    = 1
    MODE1          = 0x00
    MODE2          = 0x01
    PRE_SCALE      = 0xFE
    PWM_FREQ       = 50

    # --- Channel mapping ---
    STEER_CH       = 0
    THROTTLE_CH    = 1

    # --- Steering limits (microseconds) ---
    # NOTE: reversed — servo is physically inverted
    STEER_LEFT     = 2100
    STEER_CENTER   = 1500
    STEER_RIGHT    = 900

    # --- Throttle limits (microseconds) ---
    THROTTLE_NEUTRAL  = 1500
    THROTTLE_FWD_MAX  = 1900
    THROTTLE_REV_MAX  = 1100

    # --- Lidar config ---
    LIDAR_PORT     = '/dev/ttyUSB0'
    LIDAR_BAUD     = 115200

    def __init__(self, init_lidar=False):
        self.bus   = None
        self.lidar = None
        self._init_pca9685()
        if init_lidar:
            self._init_lidar()

    # ------------------------------------------------
    # PCA9685 internal
    # ------------------------------------------------

    def _init_pca9685(self):
        self.bus = smbus.SMBus(self.PCA9685_BUS)
        pre  = round(25_000_000 / (4096 * self.PWM_FREQ)) - 1
        old  = self._read_reg(self.MODE1)
        self._write_reg(self.MODE1, (old & 0x7F) | 0x10)
        self._write_reg(self.PRE_SCALE, pre)
        self._write_reg(self.MODE1, old)
        time.sleep(0.005)
        self._write_reg(self.MODE1, old | 0xA0)
        print(f"PCA9685 ready — PRE_SCALE={pre:#x} ({self.PWM_FREQ}Hz)")

    def _write_reg(self, reg, val):
        self.bus.write_byte_data(self.PCA9685_ADDR, reg, val)

    def _read_reg(self, reg):
        return self.bus.read_byte_data(self.PCA9685_ADDR, reg)

    def _set_us(self, channel, us):
        count = int(us / (1_000_000 / self.PWM_FREQ) * 4096)
        base  = 0x06 + channel * 4
        self.bus.write_i2c_block_data(self.PCA9685_ADDR, base, [
            0, 0, count & 0xFF, count >> 8
        ])

    # ------------------------------------------------
    # Steering
    # ------------------------------------------------

    def steer(self, val):
        """val: -1.0 = full left, 0.0 = center, 1.0 = full right"""
        val = max(-1.0, min(1.0, val))
        if val >= 0:
            us = self.STEER_CENTER + val * (self.STEER_RIGHT - self.STEER_CENTER)
        else:
            us = self.STEER_CENTER + val * (self.STEER_CENTER - self.STEER_LEFT)
        self._set_us(self.STEER_CH, us)
        print(f"Steer {val:+.2f} → {us:.0f}us")

    def steer_left(self, amount=1.0):
        self.steer(-abs(amount))

    def steer_right(self, amount=1.0):
        self.steer(abs(amount))

    def steer_center(self):
        self.steer(0.0)

    # ------------------------------------------------
    # Throttle
    # ------------------------------------------------

    def throttle(self, val):
        """val: -1.0 = full reverse, 0.0 = neutral, 1.0 = full forward"""
        val = max(-1.0, min(1.0, val))
        if val >= 0:
            us = self.THROTTLE_NEUTRAL + val * (self.THROTTLE_FWD_MAX - self.THROTTLE_NEUTRAL)
        else:
            us = self.THROTTLE_NEUTRAL + val * (self.THROTTLE_NEUTRAL - self.THROTTLE_REV_MAX)
        self._set_us(self.THROTTLE_CH, us)
        print(f"Throttle {val:+.2f} → {us:.0f}us")

    def forward(self, amount=0.2):
        self.throttle(abs(amount))

    def reverse(self, amount=0.2):
        self.throttle(-abs(amount))

    def stop(self):
        self._set_us(self.THROTTLE_CH, self.THROTTLE_NEUTRAL)
        self._set_us(self.STEER_CH, self.STEER_CENTER)
        print("Stopped")

    # ------------------------------------------------
    # ESC arming
    # ------------------------------------------------

    def arm(self, delay=3):
        """Send neutral to all channels and wait for ESC to arm."""
        print(f"Arming ESC — waiting {delay}s...")
        for ch in range(16):
            self._set_us(ch, 1500)
        time.sleep(delay)
        print("ESC armed ✓")

    # ------------------------------------------------
    # Lidar internal
    # ------------------------------------------------

    def _init_lidar(self):
        print("Connecting to lidar...")
        self.lidar = serial.Serial(self.LIDAR_PORT, self.LIDAR_BAUD, timeout=2)
        time.sleep(0.5)
        self.lidar.dtr = False
        print("Motor spinning — waiting 3s...")
        time.sleep(3)
        self._lidar_reset()
        print("Lidar ready ✓")

    def _lidar_reset(self):
        self.lidar.reset_input_buffer()
        self.lidar.write(b'\xA5\x40')
        time.sleep(2)
        self.lidar.reset_input_buffer()

    def _lidar_stop_scan(self):
        self.lidar.write(b'\xA5\x25')
        time.sleep(0.1)
        self.lidar.reset_input_buffer()

    # ------------------------------------------------
    # Lidar health
    # ------------------------------------------------

    def lidar_health(self):
        if not self.lidar:
            print("Lidar not initialized — use JetRacer(init_lidar=True)")
            return None
        self.lidar.reset_input_buffer()
        self.lidar.write(b'\xA5\x52')
        time.sleep(0.2)
        health = self.lidar.read(10)
        self.lidar.reset_input_buffer()
        if len(health) >= 8:
            status = health[7]
            labels = {0: "GOOD", 1: "WARNING", 2: "ERROR"}
            result = labels.get(status, "UNKNOWN")
            print(f"Lidar health: {result}")
            return result
        return None

    # ------------------------------------------------
    # Lidar scan
    # ------------------------------------------------

    def lidar_scan(self, samples=360):
        """
        Returns dict of {angle_deg: distance_mm}
        Performs a clean reset before every scan.
        """
        if not self.lidar:
            print("Lidar not initialized — use JetRacer(init_lidar=True)")
            return {}

        self._lidar_stop_scan()
        self._lidar_reset()

        self.lidar.reset_input_buffer()
        self.lidar.write(b'\xA5\x20')
        time.sleep(0.5)
        self.lidar.read(7)

        readings = {}
        errors   = 0

        while len(readings) < samples:
            raw = self.lidar.read(5)
            if len(raw) < 5:
                errors += 1
                if errors > 200:
                    print("Too many read errors — stopping scan")
                    break
                continue

            b0, b1, b2, b3, b4 = raw
            quality  = b0 >> 2
            angle    = ((b2 << 7) | (b1 >> 1)) / 64.0
            distance = ((b4 << 8) | b3) / 4.0

            if quality > 0 and distance > 0:
                sector = int(angle)
                if sector not in readings:
                    readings[sector] = distance
            else:
                errors += 1
                if errors > 200:
                    print("Too many zero readings — lidar may not be spinning")
                    break

        self._lidar_stop_scan()
        return readings

    # ------------------------------------------------
    # Lidar helpers
    # ------------------------------------------------

    def lidar_closest(self):
        readings = self.lidar_scan(samples=360)
        if not readings:
            return None, None
        angle = min(readings, key=readings.get)
        dist  = readings[angle]
        print(f"Closest: {dist:.1f}mm at {angle}°")
        return angle, dist

    def lidar_summary(self):
        readings = self.lidar_scan(samples=360)
        if not readings:
            print("No lidar readings")
            return
        distances = list(readings.values())
        print(f"Samples  : {len(readings)}")
        print(f"Closest  : {min(distances):.1f}mm at {min(readings, key=readings.get)}°")
        print(f"Farthest : {max(distances):.1f}mm at {max(readings, key=readings.get)}°")
        print(f"Average  : {sum(distances)/len(distances):.1f}mm")

    # ------------------------------------------------
    # Cleanup
    # ------------------------------------------------

    def close(self):
        self.stop()
        if self.lidar:
            self._lidar_stop_scan()
            self.lidar.dtr = True
            self.lidar.close()
        if self.bus:
            self.bus.close()
        print("JetRacer closed")