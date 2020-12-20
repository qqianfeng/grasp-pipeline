#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from geometry_msgs.msg import Pose, Quaternion
from sensor_msgs.msg import JointState
import moveit_msgs.msg
import moveit_commander
import numpy as np
from trac_ik_python.trac_ik import IK

DEBUG = False


class CartesianPoseMoveitPlanner():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moveit_goal_pose_planner_node')

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander('panda_arm')

        self.group.set_planner_id('RRTConnectConfigDefault')
        self.group.set_planning_time(10)
        self.group.set_num_planning_attempts(3)

        self.zero_joint_states = np.zeros(7)
        self.home_joint_states = np.array([0, -0.553876, 0, -2.5361, 0, 1.9884, -0.785])

        self.ik_solver = IK("world", "panda_link8")  #panda_link0 was world before
        print(self.ik_solver.link_names)
        print(self.ik_solver.joint_names)
        print(self.ik_solver.number_of_joints)
        self.seed_state = [0.0] * self.ik_solver.number_of_joints

    def go_home(self):
        print 'go home'
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(self.home_joint_states)
        plan_home = self.group.plan()
        return plan_home

    def go_zero(self):
        print 'go zero'
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(self.zero_joint_states)
        plan_home = self.group.plan()
        return plan_home

    def go_goal(self, pose):
        print 'go goal'
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(pose)
        plan_goal = self.group.plan()
        return plan_goal

    def go_goal_trac_ik(self, pose):
        print 'go goal trac ik'
        self.group.clear_pose_targets()
        ik_js = self.ik_solver.get_ik(self.seed_state, pose.position.x, pose.position.y,
                                      pose.position.z, pose.orientation.x, pose.orientation.y,
                                      pose.orientation.z, pose.orientation.w)
        #ik_js = self.ik_solver.get_ik(self.seed_state, 0.45, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0)
        if ik_js is None:
            rospy.logerr('No IK solution found')
            return None
        self.group.set_joint_value_target(np.array(ik_js))
        plan_goal = self.group.plan()
        return plan_goal

    def handle_arm_moveit_cartesian_pose_planner(self, req):
        #panda_joints = rospy.wait_for_message('panda/joint_states', JointState)
        #self.seed_state = list(panda_joints.position)
        plan = None
        if req.go_home:
            plan = self.go_home()
        elif req.go_zero:
            plan = self.go_zero()
        else:
            plan = self.go_goal_trac_ik(req.palm_goal_pose_world)
        response = PalmGoalPoseWorldResponse()
        response.success = False
        if plan is None:
            return response
        if len(plan.joint_trajectory.points) > 0:
            response.success = True
            response.plan_traj = plan.joint_trajectory
        return response

    def create_arm_moveit_cartesian_pose_planner_server(self):
        rospy.Service('arm_moveit_cartesian_pose_planner', PalmGoalPoseWorld,
                      self.handle_arm_moveit_cartesian_pose_planner)
        rospy.loginfo('Service arm_moveit_cartesian_pose_planner:')
        rospy.loginfo('Reference frame: %s' % self.group.get_planning_frame())
        rospy.loginfo('End-effector frame: %s' % self.group.get_end_effector_link())
        rospy.loginfo('Robot Groups: %s' % self.robot.get_group_names())
        rospy.loginfo('Ready to start to plan for given palm goal poses.')

    def handle_arm_movement(self, req):
        self.group.go(wait=True)
        response = MoveArmResponse()
        response.success = True
        return response

    def create_arm_movement_server(self):
        rospy.Service('arm_movement', MoveArm, self.handle_arm_movement)
        rospy.loginfo('Service moveit_cartesian_pose_planner:')
        rospy.loginfo('Ready to start to execute movement plan on robot arm.')


if __name__ == '__main__':
    planner = CartesianPoseMoveitPlanner()
    if DEBUG:
        gp = Pose()
        gp.position.x, gp.position.y, gp.position.z = 0.100, -0.377, 0.457
        gp.orientation.x, gp.orientation.y, gp.orientation.z, gp.orientation.w = 0.748, 0.663, -0.001, -0.000
        req = PalmGoalPoseWorldRequest()
        req.go_home = False
        req.go_zero = False
        req.palm_goal_pose_world = gp
        planner.handle_arm_moveit_cartesian_pose_planner(req)
    planner.create_arm_moveit_cartesian_pose_planner_server()
    rospy.spin()
