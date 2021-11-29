
import rospy
import geometry_msgs.msg
import perception.msg

import numpy as np

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

class EkfDataSaver(GenericOutputSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, perception.msg.EkfOutput,
            self._ekf_output_cb
        )

    def _ekf_output_cb(self, msg):

        # Rotate x,y,z coordinates around z-axis to align body frame with NED
        Rz = np.array([
            [np.cos(msg.psi), -np.sin(msg.psi), 0],
            [np.sin(msg.psi), np.cos(msg.psi), 0],
            [0, 0, 1]
        ])

        pos_body = np.array([msg.x, msg.y, msg.z])

        n,e,d = Rz @ pos_body

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
