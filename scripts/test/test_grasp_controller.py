#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState
import numpy as np


class Test():
    def __init__(self):
        rospy.init_node('test_listener')
        hithand_joints = rospy.wait_for_message("/hithand/joint_states", JointState, 1)
        pos = hithand_joints.position
        pos_np = np.array(pos)
        self.delta_pos_joint_1 = 2  # 2rad/s, when sending at 100 Hz
        self.delta_pos_joint_2 = 1  # 1rad/s, when sending at 100 Hz
        self.delta_pos_joint_3 = 0.5  # 0.5rad/s, when sending at 100 Hz
        self.delta_vector = np.array([
            0, self.delta_pos_joint_1, self.delta_pos_joint_2, self.delta_pos_joint_3
        ])
        self.delta_matrix = np.tile(self.delta_vector, 5)
        pos_np += self.delta_matrix
        print(pos_np)


if __name__ == '__main__':
    t = Test()