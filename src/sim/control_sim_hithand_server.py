#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Quaternion
import numpy as np
import moveit_msgs.msg


class ControlHithandConfig():
    def __init__(self, publish_prefix='/hithand'):
        rospy.init_node('control_hithand_config_node')
        self.hithand_joint_cmd_pub = rospy.Publisher(
            publish_prefix+'/joint_cmd', JointState, self.get_hithand_joint_state_cb)
        self.hithand_init_joint_state = None
        self.hithand_joint_state = None
        self.run_rate = rospy.Rate(20)
        #self.control_hithand_steps = 50
        #self.reach_gap_thresh = 0.1
        self.dof = 20

    def get_hithand_joint_state_cb(self, hithand_joint_state):
        if self.hithand_init_joint_state is None:
            self.hithand_init_joint_state = hithand_joint_state
        self.hithand_joint_state = hithand_joint_state

    def handle_control_hithand(self):
        pass

    def create_control_hithand_server(self):
        control_hithand = rospy.Service(
            'control_hithand_config', HithandControl, self.handle_control_hithand)


if __name__ == '__main__':
    control_hithand = ControlHithandConfig()
    control_hithand.create_control_hithand_server()
    rospy.spin()
