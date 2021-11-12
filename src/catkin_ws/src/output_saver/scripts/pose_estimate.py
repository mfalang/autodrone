
import rospy
import geometry_msgs.msg

from generic_output_saver import GenericOutputSaver

class DNNCVDataSaver(GenericOutputSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, geometry_msgs.msg.TwistStamped, self._dnn_cv_pose_cb)

    def _dnn_cv_pose_cb(self, msg):

        self._save_output([
            msg.header.stamp.to_sec(),
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.z
        ])
