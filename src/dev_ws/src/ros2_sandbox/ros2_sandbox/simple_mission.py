import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import olympe
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing

DRONE_IP = "10.202.0.1"

class SimpleMission(Node):

    def __init__(self):
        super().__init__("simple_mission")
        self.subscription = self.create_subscription(
            Bool,
            "start_mission",
            self.start_mission_cb,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info("Waiting for start signal")

    def start_mission_cb(self, msg):
        if msg.data:
            self.start_mission()

    def start_mission(self):
        self.get_logger().info("Starting mission")
        
        drone = olympe.Drone(DRONE_IP)
        drone.connect()
        assert drone(TakeOff()).wait().success()
        time.sleep(5)
        assert drone(Landing()).wait().success()
        drone.disconnect()

        self.get_logger().info("Mission finished")
        self.destroy_node()
        rclpy.shutdown()
        
def main(args=None):
    rclpy.init(args=args)

    simple_mission = SimpleMission()

    rclpy.spin(simple_mission)

    simple_mission.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
