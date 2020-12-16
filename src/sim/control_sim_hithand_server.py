#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Quaternion
import numpy as np
import moveit_msgs.msg

# IDEA FOR LATER: Find a low-dimensional manifold for optimal hithand grasp, learn the manifold


class ControlHithandConfig():
    def __init__(self, publish_prefix='/hithand'):
        rospy.init_node('control_hithand_config_node')
        self.hithand_joint_cmd_pub = rospy.Publisher(publish_prefix +
                                                     '/joint_cmd',
                                                     JointState,
                                                     queue_size=1)
        rospy.Subscriber('hithand/joint_states', JointState,
                         self.get_hithand_current_joint_state_cb)
        self.hithand_init_joint_state = None
        self.hithand_current_joint_state = None
        self.hithand_target_joint_state = None
        self.run_rate = rospy.Rate(20)
        self.control_hithand_steps = 50
        self.reach_gap_thresh = 0.1
        self.dof = 20

    def get_hithand_current_joint_state_cb(self, hithand_joint_state):
        if self.hithand_init_joint_state is None:
            self.hithand_init_joint_state = hithand_joint_state
        self.hithand_current_joint_state = hithand_joint_state

    def control_hithand(self):
        self.run_rate.sleep()
        rospy.loginfo('Service control_hithand_config:')
        rospy.loginfo('Joint control command published')
        jc = JointState()
        jc.name = self.hithand_target_joint_state.name
        target_joint_angles = np.array(
            self.hithand_target_joint_state.position)
        init_joint_angles = np.array(self.hithand_init_joint_state.position)
        delta = (target_joint_angles - init_joint_angles) / \
            self.control_hithand_steps
        jc_angles = init_joint_angles
        for i in xrange(self.control_hithand_steps):
            jc_angles += delta
            #print i, jc_angles
            jc.position = jc_angles.tolist()
            self.hithand_joint_cmd_pub.publish(jc)
            self.run_rate.sleep()

    def control_hithand_home(self):
        self.run_rate.sleep()
        rospy.loginfo('Service control_hithand_config:')
        rospy.loginfo('Joint control command published')

        jc = JointState()
        jc.name = self.hithand_init_joint_state.name
        target_joint_angles = np.zeros(self.dof)
        init_joint_angles = np.array(self.hithand_init_joint_state.position)
        delta = (target_joint_angles - init_joint_angles) / \
            self.control_hithand_steps
        jc_angles = init_joint_angles
        for i in xrange(self.control_hithand_steps):
            jc_angles += delta
            jc.position = jc_angles.tolist()
            self.hithand_joint_cmd_pub.publish(jc)
            self.run_rate.sleep()

    def reach_goal(self):
        reach_gap = np.array(self.hithand_target_joint_state.position) - \
            np.array(self.hithand_current_joint_state.position)
        rospy.loginfo('Service control_hithand_config:')
        rospy.loginfo('reach_gap: ')
        rospy.loginfo(str(reach_gap))
        return np.min(np.abs(reach_gap)) < self.reach_gap_thresh

    def handle_control_hithand(self, req):
        res = ControlHithandResponse()
        if req.go_home:
            self.control_hithand_home()
            res.success = True
        else:
            self.hithand_target_joint_state = req.hithand_target_joint_state
            self.control_hithand()
            res.success = self.reach_goal()
        self.hithand_init_joint_state = None
        return res

    def create_control_hithand_config_server(self):
        rospy.Service('control_hithand_config', ControlHithand,
                      self.handle_control_hithand)
        rospy.loginfo('Service control_hithand_config:')
        rospy.loginfo('Ready to control hithand to speficied configurations.')


if __name__ == '__main__':
    chc = ControlHithandConfig()
    chc.create_control_hithand_config_server()
    rospy.spin()
