#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState
import numpy as np
from copy import deepcopy
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
# The goal here is to emulate a velocity controller via a pos interface to the hithand. The goal for the hithand is to close at a constant velocity, but it should stop when the velocity is zero.
# Nevertheless it should maintain a certain force on the object.

DEBUG = True


class HithandGraspController():
    def __init__(self):
        rospy.init_node('grasp_control_hithand_node')
        self.delta_joint_angle = None
        self.freq = 100
        self.rate = rospy.Rate(self.freq)
        self.joint_command_pub = rospy.Publisher('/hithand/joint_cmd', JointState, queue_size=1)

        self.current_pos_joints = None
        self.joint_speed_rad_per_sec = 1.
        self.delta_pos_joint_1 = self.joint_speed_rad_per_sec / self.freq  # 2rad/s, when sending at 100 Hz
        self.delta_pos_joint_2 = self.joint_speed_rad_per_sec / self.freq  # 1rad/s, when sending at 100 Hz
        self.delta_pos_joint_3 = self.joint_speed_rad_per_sec / self.freq  # 0.5rad/s, when sending at 100 Hz
        self.delta_pos_joint_vector = None

        self.check_delta_joint_interval = 1

        self.pos_thresh = 0.5 * self.check_delta_joint_interval * self.delta_pos_joint_1  # If a joint moved less than this, it is considered to have zero velocity

    def init_delta_joint_vector(self):
        self.delta_vector = np.array([
            0, self.delta_pos_joint_1, self.delta_pos_joint_2, self.delta_pos_joint_3
        ])
        self.delta_pos_joint_vector = np.tile(self.delta_vector, 5)

    def update_current_pos_joints(self):
        joint_state = rospy.wait_for_message('/hithand/joint_states', JointState, 1)
        if joint_state == None:
            rospy.logerr('No hithand joint state has been received')
        self.current_pos_joints = np.array(joint_state.position)

    def handle_grasp_control_hithand(self, req):
        self.init_delta_joint_vector()
        print(self.delta_pos_joint_vector)

        self.update_current_pos_joints()
        joint_pub_msg = JointState()
        previous_pos_joints = deepcopy(self.current_pos_joints)
        desired_pos_joints = deepcopy(previous_pos_joints)
        for i in xrange(10 * self.freq):  # this loop runs for 10 seconds max
            print(i)
            if (i % self.check_delta_joint_interval == 0) and (i != 0):
                self.update_current_pos_joints()
                # check whether the joint position a few steps ago actually differs from the current joint position. If not velocity is zero and joint should stop moving
                diff_joint_pos = np.abs(self.current_pos_joints - previous_pos_joints)
                # set the deltas to zero for the joints with zero velocity in order to not send more position commands to these joints
                print(self.delta_pos_joint_vector)
                self.delta_pos_joint_vector[diff_joint_pos < self.pos_thresh] = 0
                print(self.delta_pos_joint_vector)
                # If the whole delta pos joint vector is zero break the loop, because no joints are moving anymore
                if sum(self.delta_pos_joint_vector) == 0:
                    break
                # Update the previous pos joint variable to now be the current joint position
                previous_pos_joints = self.current_pos_joints
            desired_pos_joints += self.delta_pos_joint_vector
            joint_pub_msg.position = desired_pos_joints.tolist()
            self.joint_command_pub.publish(joint_pub_msg)
            self.rate.sleep()
        res = GraspControlResponse()
        res.success = True
        return res

    def handle_reset_hithand_joints(self, req):
        reset_state = JointState()
        reset_state.position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.26, 0, 0, 0]
        self.joint_command_pub.publish(reset_state)
        res = SetBoolResponse()
        res.success = True
        return res

    def create_grasp_control_hithand_server(self):
        rospy.Service('grasp_control_hithand', GraspControl, self.handle_grasp_control_hithand)
        rospy.loginfo('Service grasp_control_hithand:')
        rospy.loginfo('Ready to control the hithand for grasping.')

    def create_reset_hithand_joints_server(self):
        rospy.Service('reset_hithand_joints', SetBool, self.handle_reset_hithand_joints)
        rospy.loginfo('Service reset_hithand_joints')
        rospy.loginfo('Ready to reset the hithand joint states.')


if __name__ == "__main__":
    hgc = HithandGraspController()
    if DEBUG:
        #hgc.handle_reset_hithand_joints(SetBoolRequest())
        hgc.handle_grasp_control_hithand(GraspControlRequest())
    # hgc.create_grasp_control_hithand_server()
    # hgc.create_reset_hithand_joints_server()
    # rospy.spin()
