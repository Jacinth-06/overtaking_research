from .racecar import Racecar
import traitlets
from adafruit_servokit import ServoKit


class NvidiaRacecar(Racecar):

    i2c_address1 = traitlets.Integer(default_value=0x40)
    i2c_address2 = traitlets.Integer(default_value=0x60)
    steering_gain = traitlets.Float(default_value=-0.65)
    steering_offset = traitlets.Float(default_value=0.0)
    steering_channel = traitlets.Integer(default_value=0)
    throttle_gain = traitlets.Float(default_value=0.0)

    def __init__(self, *args, **kwargs):
        super(NvidiaRacecar, self).__init__(*args, **kwargs)

        # Initialize PCA9685 boards
        self.kit = ServoKit(channels=16, address=self.i2c_address1)
        self.motor = ServoKit(channels=16, address=self.i2c_address2)

        # Set motor PWM frequency
        self.motor._pca.frequency = 1600

        # HARD STOP ALL CHANNELS (CRITICAL)
        for ch in range(16):
            self.motor._pca.channels[ch].duty_cycle = 0

        # Steering servo
        self.steering_motor = self.kit.continuous_servo[self.steering_channel]
        self.steering_motor.throttle = 0.0

        # Initialize trait values AFTER hardware stop
        self.throttle = 0.0
        self.steering = 0.0

    @traitlets.observe('steering')
    def _on_steering(self, change):
        self.steering_motor.throttle = (
            change['new'] * self.steering_gain + self.steering_offset
        )

    @traitlets.observe('throttle')
    def _on_throttle(self, change):
        t = change['new'] * self.throttle_gain

        # DEADZONE + STOP
        if abs(t) < 0.05:
            for ch in range(16):
                self.motor._pca.channels[ch].duty_cycle = 0
            return

        # FORWARD
        if t > 0:
            self.motor._pca.channels[0].duty_cycle = int(0xFFFF * t)
            self.motor._pca.channels[1].duty_cycle = 0xFFFF
            self.motor._pca.channels[2].duty_cycle = 0
            self.motor._pca.channels[3].duty_cycle = 0

            self.motor._pca.channels[4].duty_cycle = int(0xFFFF * t)
            self.motor._pca.channels[7].duty_cycle = int(0xFFFF * t)
            self.motor._pca.channels[6].duty_cycle = 0xFFFF
            self.motor._pca.channels[5].duty_cycle = 0

        # REVERSE
        else:
            t = abs(t)

            self.motor._pca.channels[0].duty_cycle = int(0xFFFF * t)
            self.motor._pca.channels[1].duty_cycle = 0
            self.motor._pca.channels[2].duty_cycle = 0xFFFF
            self.motor._pca.channels[3].duty_cycle = 0

            self.motor._pca.channels[4].duty_cycle = int(0xFFFF * t)
            self.motor._pca.channels[7].duty_cycle = int(0xFFFF * t)
            self.motor._pca.channels[6].duty_cycle = 0
            self.motor._pca.channels[5].duty_cycle = 0xFFFF

