#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from grasp_pipeline.srv import *
from trajectory_smoothing.srv import *
import numpy as np
import time


class RobotTrajectoryManager():
    def __init__(self):
        self.joint_command_pub = rospy.Publisher('panda/joint_cmd', JointState, queue_size=1)
        self.joint_trajectory = None
        self.rate_hz = 100
        self.dt = 1 / self.rate_hz
        self.loop_rate = rospy.Rate(self.rate_hz)
        self.max_acceleration = 0.25 * np.ones(7)
        self.max_velocity = 0.4 * np.ones(7)

    def get_smooth_trajectory_client(self):
        rospy.loginfo('Waiting for service get_smooth_trajectory.')
        rospy.wait_for_service('/get_smooth_trajectory')
        rospy.loginfo('Calling service get_smooth_trajectory.')
        try:
            smoothen_trajectory = rospy.ServiceProxy('/get_smooth_trajectory', GetSmoothTraj)
            req = GetSmoothTrajRequest()
            req.init_traj = self.joint_trajectory
            req.max_acc = self.max_acceleration
            req.max_vel = self.max_velocity
            req.max_dev = 0.1
            req.dt = self.dt
            res = smoothen_trajectory(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service get_smooth_trajectory call failed: %s' % e)
        self.joint_trajectory = res.smooth_traj  # Joint trajectory is now smooth
        rospy.loginfo('Service get_smooth_trajectory is executed %s.' % str(res.success))

    def send_trajectory(self):
        for i in range(len(self.joint_trajectory.points)):
            new_jc = JointState()
            new_jc.name = self.joint_trajectory.joint_names
            new_jc.position = self.joint_trajectory.points[i].positions
            new_jc.velocity = self.joint_trajectory.points[i].velocities
            new_jc.effort = self.joint_trajectory.points[i].accelerations
            self.joint_command_pub.publish(new_jc)
            self.loop_rate.sleep()

    def handle_execute_joint_trajectory(self, req):
        self.joint_trajectory = req.joint_trajectory
        if req.smoothen_trajectory:
            self.get_smooth_trajectory_client()

        self.send_trajectory()

        desired_joint_state = self.joint_trajectory.points[-1].positions
        reached = False

        start_time = time.time()
        # Check if goal is reached:
        while (reached == False):
            if time.time() - start_time >= 2:
                break
            arm_joint_state = rospy.wait_for_message('/panda/joint_states', JointState)
            err = np.linalg.norm(
                np.array(desired_joint_state) -
                np.array(arm_joint_state.position))  # ARM JOINT STATE FEHLT HIER!!!
            if (err < 0.01):
                reached = True
            self.loop_rate.sleep()
        rospy.loginfo('***Arm reached: %s' % str(reached))
        rospy.loginfo('***Arm reach error: %s' % str(err))

    def create_execute_joint_trajectory_server(self):
        rospy.Service('execute_joint_trajectory', ExecuteJointTrajectory,
                      self.handle_execute_joint_trajectory)
        rospy.loginfo('Service execute_joint_trajectory:')
        rospy.loginfo('Ready to execute joint trajectory on robot arm.')


if __name__ == '__main__':
    rti = RobotTrajectoryManager()
    rti.create_execute_joint_trajectory_server()
    rospy.spin()