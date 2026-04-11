from jetracer import JetRacer
import time

car = JetRacer()
car.arm(delay=3)

try:
    print("Steering left...")
    car.steer_left(0.5)
    time.sleep(2)

    print("Steering right...")
    car.steer_right(0.5)
    time.sleep(2)

    print("Center...")
    car.steer_center()
    time.sleep(1)

    print("Slow forward...")
    car.forward(0.1)
    time.sleep(2)

    print("Forward + steer left...")
    car.steer_left(0.4)
    time.sleep(2)

    print("Forward + steer right...")
    car.steer_right(0.4)
    time.sleep(2)

    print("Stopping...")
    car.stop()

except KeyboardInterrupt:
    print("Interrupted")
    car.stop()

finally:
    car.close()