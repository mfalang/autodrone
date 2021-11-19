
import ekf

import os
import sys
import yaml

import rospy
import geometry_msgs.msg
import pose_estimate.msg
import numpy as np

from ekf.ekf import EKF, EKFState
from ekf.dynamic_models.helipad_3D_movement import DynamicModel
from ekf.measurement_models.dnn_cv import DnnCvModel

class EKFRosRunner():

    def __init__(self):
        
        rospy.init_node("perception_ekf", anonymous=False)

        config_file = rospy.get_param("~config_file")
        script_dir = os.path.dirname(os.path.realpath(__file__))

        try:
            with open(f"{script_dir}/../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()    
    
        self.z = np.zeros((4))
        self.new_messurement = False
        
        self.dt = self.config["ekf"]["dt"]

        self.estimate_publisher = rospy.publisher(
            self.config["ekf"]["output"]["topic_name"], 
            geometry_msgs.msg.EkfOutput, queue_size=10
        )


    def run(self):
        rospy.loginfo("Starting perception EKF")
        
        filter = EKF(DynamicModel(), DnnCvModel())

        x0 = np.asarry(self.config["ekf"]["init_values"]["x0"])
        P0 = np.asarry(self.config["ekf"]["init_values"]["P0"])

        ekf_estimate = EKFState(x0, P0)

        while not rospy.is_shutdown():
            
            ekf_estimate = filter.predict(ekf_estimate, self.dt)

            if self.new_messurement:
                ekf_estimate = filter.update(self.z, ekf_estimate)
                self.new_messurement = False

            output_msg = self._pack_estimate_msg(self, ekf_estimate)
            self.estimate_publisher.publish(output_msg)

    def _pack_estimate_msg(self, ekfstate):
        msg = pose_estimate.msg.EkfOutput()
        msg.header.stamp = rospy.time.Now()
        msg.x = ekfstate.mean[0]
        msg.y = ekfstate.mean[1]    
        msg.z = ekfstate.mean[2]
        msg.psi = ekfstate.mean[3]    
        msg.v_x = ekfstate.mean[4]    
        msg.v_y = ekfstate.mean[5]    
        msg.v_z = ekfstate.mean[6]    
        msg.covariance = ekfstate.cov.flatten().tolist()

        return msg

    def _dnn_cv_estimate_cb(self, msg):
        self.z = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.z
        ])
        self.new_messurement = True

def main():
    ekf_runner = EKFRosRunner()
    ekf_runner.run()

if __name__ == "__main__":
    main()