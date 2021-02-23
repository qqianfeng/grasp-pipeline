#!/usr/bin/env python
import rospy
import time
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
import numpy as np
from copy import deepcopy
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
# The goal here is to emulate a velocity controller via a pos interface to the hithand. The goal for the hithand is to close at a constant velocity, but it should stop when the velocity is zero.
# Nevertheless it should maintain a certain force on the object.


class HithandGraspController():
    def __init__(self):
        rospy.init_node('grasp_control_hithand_node')
        self.pub_freq = 100
        self.rate = rospy.Rate(self.pub_freq)
        self.joint_command_pub = rospy.Publisher('/hithand/joint_cmd', JointState, queue_size=1)
        self.start_reset_cond = True
        # self.start_reset_sub = rospy.Subscriber("/start_hithand_reset",
        #                                         Bool,
        #                                         callback=self.cb_start_reset_pub,
        #                                         queue_size=1)

        self.curr_pos = None
        self.received_curr_pos = False
        self.joint_speed_rad_per_sec = 0.5
        self.delta_pos_joint_1 = self.joint_speed_rad_per_sec / self.pub_freq
        self.delta_pos_joint_2 = self.joint_speed_rad_per_sec / self.pub_freq
        self.delta_pos_joint_3 = self.joint_speed_rad_per_sec / self.pub_freq
        self.delta_vector = np.array([
            0, self.delta_pos_joint_1, self.delta_pos_joint_2, self.delta_pos_joint_3
        ])

        self.check_vel_interval = 15
        self.vel_thresh = 1e-2  # In % of expected movement. If a joint moved less than this, it is considered to have zero velocity
        self.avg_vel = None
        self.moving_avg_vel = np.zeros([
            20, self.check_vel_interval
        ])  # Rows is vel of each joint for one measurement, cols are successive measurements
        self.check_vel_thumb = 20
        self.moving_avg_thmub = np.zeros([4, self.check_vel_thumb])
        self.reset_position = [
            0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ]

        # For controlling hithand config
        self.run_rate = rospy.Rate(50)
        self.control_steps = 50
        self.reach_gap_thresh = 0.1

        self.thumb_0_max_torque = 1.15

        self.max_torque_cnt = 8

    def verify_needs_reset(self):
        while True:
            if self.received_curr_pos:
                break
        pos_diff = np.abs(np.array(self.reset_position) - self.curr_pos)
        # If at least one joint is more than 1e-3 away from where it's supposed to be, say hithand needs reset
        if pos_diff[pos_diff > 8e-3].size == 0:
            return False
        else:
            return True

    def get_delta_joint_vector(self):
        delta_vector = np.tile(self.delta_vector, 5)
        delta_vector[-4:] = 2 * delta_vector[-4:]
        return np.tile(self.delta_vector, 5)

    def cb_start_reset_pub(self, msg):
        print("Start reset pub callback entered")
        if msg.data == self.start_reset_cond:
            print("Start reset pub callback EXECUTE")
            self.start_reset_cond = not self.start_reset_cond
            joint_states_sub = rospy.Subscriber(
                '/hithand/joint_states', JointState, self.cb_update_curr_pos, tcp_nodelay=True
            )  # Setting tcp_nodelay true is crucial otherwise the callback won't be executed at 100Hz
            needs_reset = self.verify_needs_reset()
            # Only command reset position if not in reset position already
            if needs_reset:
                reset_state = JointState()
                reset_state.position = self.reset_position
                start = time.time()
                while self.verify_needs_reset() and (time.time() - start) < 5:
                    self.joint_command_pub.publish(reset_state)
                    rospy.sleep(0.1)

            joint_states_sub.unregister()

    def cb_update_curr_pos(self, msg):
        #start = time.time()
        self.curr_pos = np.array(msg.position)
        self.curr_vel = np.abs(np.array(msg.velocity))
        self.received_curr_pos = True

        # Compute moving avg of last self.check_vel_interval velocity measurements
        self.moving_avg_vel[:, 1:] = self.moving_avg_vel[:, :-1]  # shift all columns right
        self.moving_avg_vel[:, 0] = self.curr_vel
        self.avg_vel = np.sum(self.moving_avg_vel, 1, dtype=np.float) / self.check_vel_interval

        self.thumb_effort = np.array(msg.effort)[-4:]

    def handle_grasp_control_hithand(self, req):
        # First subscribe to the hithand joint state topic, which will continuously update the joint position at 100Hz
        joint_states_sub = rospy.Subscriber(
            '/hithand/joint_states', JointState, self.cb_update_curr_pos, tcp_nodelay=True
        )  # Setting tcp_nodelay true is crucial otherwise the callback won't be executed at 100Hz
        while True:
            if self.received_curr_pos:
                break

        # Initialize the delta vector every time as this changes during execution
        delta_pos = self.get_delta_joint_vector()
        joint_pos_msg = JointState()
        desired_pos = self.curr_pos

        for i in xrange(10 * self.pub_freq):  # this loop runs for 10 seconds max
            if i > 5 and i % self.check_vel_interval == 0:
                print("Index_joint_vel AVG: " + str(self.avg_vel[:4]))
                delta_pos[self.avg_vel < self.vel_thresh] = 0
                if sum(delta_pos) == 0:
                    break
            desired_pos += delta_pos
            joint_pos_msg.position = desired_pos.tolist()
            self.joint_command_pub.publish(joint_pos_msg)
            self.rate.sleep()

        # Unregister as we don't want to continuously update the joints
        joint_states_sub.unregister()

        res = GraspControlResponse()
        res.success = True
        return res

    def handle_reset_hithand_joints(self, req):
        joint_states_sub = rospy.Subscriber('/hithand/joint_states',
                                            JointState,
                                            self.cb_update_curr_pos,
                                            tcp_nodelay=True)
        needs_reset = self.verify_needs_reset()
        # Only command reset position if not in reset position already
        if needs_reset:
            reset_state = JointState()
            reset_state.position = self.reset_position
            start = time.time()
            while self.verify_needs_reset() and (time.time() - start) < 5:
                self.joint_command_pub.publish(reset_state)
                rospy.sleep(0.1)

        joint_states_sub.unregister()

        res = SetBoolResponse()
        res.success = True
        return res

    def control_hithand(self):
        target_joint_pos = np.array(self.target_joint_state.position)

        # Bring the thumb to its initial position first:
        pos = np.zeros(20)
        pos[16] = target_joint_pos[16]
        js = JointState(position=pos)
        for i in xrange(3):
            self.joint_command_pub.publish(js)
            rospy.sleep(0.1)
        # Sleep to make sure current pos is updated
        rospy.sleep(0.05)

        jc = JointState()
        init_joint_pos = self.curr_pos
        delta = (target_joint_pos - init_joint_pos) / self.control_steps
        delta[16] = 0  # Set thumb abd joint angle delta to 0
        delta[16:] *= 1.5
        jc_pos = init_joint_pos

        max_torque_cnt = 0

        for i in xrange(self.control_steps):
            print("Thumb effort" + str(self.thumb_effort))
            if i > 5:
                #delta[self.avg_vel < self.vel_thresh] = 0
                if sum(delta) < 1e-3:  # or min(self.thumb_avg_vel[1:]) < -0.05:
                    return
                if self.thumb_effort[1] == self.thumb_0_max_torque:
                    max_torque_cnt += 1
                    if max_torque_cnt == self.max_torque_cnt:
                        return
                else:
                    max_torque_cnt = 0

            jc_pos += delta
            jc.position = jc_pos.tolist()
            self.joint_command_pub.publish(jc)
            self.run_rate.sleep()

        # If code got till here the hand moved to the full config without significant contact
        # Keep sending the position deltas until significant contact is established
        max_torque_cnt = 0
        start = time.time()
        while (max_torque_cnt != self.max_torque_cnt) and (time.time() - start < 5):
            if self.thumb_effort[1] == self.thumb_0_max_torque:
                max_torque_cnt += 1
            else:
                max_torque_cnt = 0
            jc_pos += delta
            jc.position = jc_pos.tolist()
            self.joint_command_pub.publish(jc)
            self.run_rate.sleep()

    def reach_goal(self):
        reach_gap = np.array(self.target_joint_state.position) - self.curr_pos
        rospy.loginfo('Service control_hithand_config:')
        rospy.loginfo('reach_gap: ')
        rospy.loginfo(str(reach_gap))
        return np.min(np.abs(reach_gap)) < self.reach_gap_thresh

    def handle_control_hithand(self, req):
        joint_states_sub = rospy.Subscriber('/hithand/joint_states',
                                            JointState,
                                            self.cb_update_curr_pos,
                                            tcp_nodelay=True)
        res = ControlHithandResponse()
        self.target_joint_state = req.hithand_target_joint_state
        self.control_hithand()
        res.success = self.reach_goal()
        self.init_joint_state = None

        joint_states_sub.unregister()

        return res

    def create_grasp_control_hithand_server(self):
        rospy.Service('grasp_control_hithand', GraspControl, self.handle_grasp_control_hithand)
        rospy.loginfo('Service grasp_control_hithand:')
        rospy.loginfo('Ready to control the hithand for grasping.')

    def create_reset_hithand_server(self):
        rospy.Service('reset_hithand_joints', SetBool, self.handle_reset_hithand_joints)
        rospy.loginfo('Service reset_hithand_joints')
        rospy.loginfo('Ready to reset the hithand joint states.')

    def create_control_hithand_config_server(self):
        rospy.Service('control_hithand_config', ControlHithand, self.handle_control_hithand)
        rospy.loginfo('Service control_hithand_config:')
        rospy.loginfo('Ready to control hithand to speficied configurations.')


if __name__ == "__main__":
    hgc = HithandGraspController()
    hgc.create_grasp_control_hithand_server()
    hgc.create_reset_hithand_server()
    hgc.create_control_hithand_config_server()
    rospy.spin()
