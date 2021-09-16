import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

class MissionStarter(Node):

    def __init__(self):
        super().__init__("mission_starter")
        self.publisher = self.create_publisher(Bool, "start_mission", 10)

    def check_for_keypress(self):
        key = input("Press enter to start mission ")
        while key != "":
            key = input()
        msg = Bool()
        msg.data = True
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    mission_starter = MissionStarter()
    mission_starter.check_for_keypress()

    mission_starter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
