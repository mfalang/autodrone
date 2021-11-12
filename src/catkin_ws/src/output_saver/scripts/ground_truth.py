
import rospy
import geometry_msgs.msg

from scipy.spatial.transform import Rotation

from generic_output_saver import GenericOutputSaver

class DronePoseDataSaver(GenericOutputSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, geometry_msgs.msg.PoseStamped, self._drone_gt_pose_cb)

    def _drone_gt_pose_cb(self, msg):
        self._save_output(get_output_from_geometry_msg(msg))

class HelipadPoseDataSaver(GenericOutputSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, geometry_msgs.msg.PoseStamped, self._helipad_gt_pose_cb)

    def _helipad_gt_pose_cb(self, msg):
        self._save_output(get_output_from_geometry_msg(msg))

# Helper function
def get_output_from_geometry_msg(msg):

    res = []
    res.append(msg.header.stamp.to_sec())
    res.append(msg.pose.position.x)
    res.append(msg.pose.position.y)
    res.append(msg.pose.position.z)

    quat = [msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w
    ]
    euler = Rotation.from_quat(quat).as_euler("xyz", degrees=True)
    res.append(euler[0])
    res.append(euler[1])
    res.append(euler[2])

    return res
