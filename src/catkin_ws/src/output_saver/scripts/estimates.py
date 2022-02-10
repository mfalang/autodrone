
import rospy
import geometry_msgs.msg
import perception.msg

from generic_output_saver import GenericOutputSaver

class DnnCvPositionSaver(GenericOutputSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, geometry_msgs.msg.PointStamped, self._dnn_cv_position_cb)

    def _dnn_cv_position_cb(self, msg: geometry_msgs.msg.PointStamped):

        self._save_output([
            msg.header.stamp.to_sec(),
            msg.point.x,
            msg.point.y,
            msg.point.z
        ])

class DnnCvHeadingSaver(GenericOutputSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, perception.msg.Heading, self._dnn_cv_heading_cb)

    def _dnn_cv_heading_cb(self, msg: perception.msg.Heading):

        self._save_output([
            msg.header.stamp.to_sec(),
            msg.heading
        ])

class TcvDataSaver(GenericOutputSaver):
    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, perception.msg.EulerPose, self._tcv_pose_cb)

    def _tcv_pose_cb(self, msg):

        self._save_output([
            msg.header.stamp.to_sec(),
            msg.x,
            msg.y,
            msg.z,
            msg.phi,
            msg.theta,
            msg.psi
        ])

class EkfDataSaver(GenericOutputSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, perception.msg.EkfOutput,
            self._ekf_output_cb
        )

    def _ekf_output_cb(self, msg):
        output = [
            msg.header.stamp.to_sec(),
            msg.x,
            msg.y,
            msg.z,
            msg.psi,
            msg.v_x,
            msg.v_y,
            msg.v_z
        ]

        output.extend(msg.covariance)

        self._save_output(output)
