#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from grasp_pipeline.srv import *

import numpy as np


class RobotTrajectoryManager():
    def __init__(self):
        self.joint_command_publisher = rospy.Publisher('panda/joint_cmd', JointState, queue_size=1)
        self.joint_trajectory = None
        self.smooth_joint_trajectory = None

    def smoothen_joint_trajectory_client(self):
        rospy.loginfo('Waiting for service smoothen_joint_trajectory.')
        rospy.wait_for_service('/smoothen_joint_trajectory')
        rospy.loginfo('Calling service smoothen_joint_trajectory.')
        try:
            smoothen_trajectory = rospy.ServiceProxy('/smoothen_joint_trajectory',
                                                     SmoothenJointTrajectory)
            req = SmoothenJointTrajectoryRequest()
            req.joint_trajectory = self.joint_trajectory
            res = smoothen_trajectory(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service moveit_cartesian_pose_planner call failed: %s' % e)
        self.smooth_joint_trajectory = res.smooth_joint_trajectory
        rospy.loginfo('Service moveit_cartesian_pose_planner is executed %s.' % str(res.success))

    def handle_execute_joint_trajectory(self, req):
        self.joint_trajectory = req.joint_trajectory
        if req.smoothen_trajectory:
            self.smoothen_joint_trajectory_client()

    def create_execute_joint_trajectory_server(self):
        rospy.Service('execute_joint_trajectory', ExecuteJointTrajectory,
                      self.handle_execute_joint_trajectory)
        rospy.loginfo('Service execute_joint_trajectory:')
        rospy.loginfo('Ready to execute joint trajectory on robot arm.')


if __name__ == '__main__':
    rti = RobotTrajectoryManager()
    rti.create_execute_joint_trajectory_server()
    rospy.spin()