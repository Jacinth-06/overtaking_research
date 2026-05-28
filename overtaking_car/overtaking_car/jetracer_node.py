import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


# This imports from your existing jetracer library 
# (We will make sure Python can see it via a symlink)
from overtaking_car.jetracer import JetRacer

class JetRacerHardwareNode(Node):
    def __init__(self):
        super().__init__('jetracer_hardware_node')
        self.car = JetRacer()
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.get_logger().info('JetRacer Hardware Node is active and listening to /cmd_vel')

    def cmd_vel_callback(self, msg):
        # Translate ROS 2 Twist messages to JetRacer steering/throttle
        self.car.throttle = msg.linear.x
        self.car.steering = msg.angular.z
        self.get_logger().info(f'Throttle: {msg.linear.x:.2f}, Steering: {msg.angular.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = JetRacerHardwareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.car.throttle = 0.0
        node.car.steering = 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()