#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import JointState
import moveit_msgs.msg
import moveit_commander
import numpy as np
from trac_ik_python.trac_ik import IK


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

        self.home_joint_states = np.array([0, 0, 0, -1, 0, 1.9884, -1.57])

        self.ik_solver = IK("world", "palm_link_hithand", timeout=1,
                            epsilon=1e-4)  #panda_link0 was world before
        self.seed_state = [0.0] * self.ik_solver.number_of_joints

        self.solver_margin_pos = 0.01
        self.solver_margin_ori = 0.05

    def go_home(self):
        print 'go home'
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(self.home_joint_states)
        plan_home = self.group.plan()
        return plan_home

    def go_goal_trac_ik(self, pose):
        print 'go goal trac ik'
        self.group.clear_pose_targets()
        ik_js = self.ik_solver.get_ik(
            self.seed_state, pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
            pose.pose.orientation.w, self.solver_margin_pos, self.solver_margin_pos,
            self.solver_margin_pos, self.solver_margin_ori, self.solver_margin_ori,
            self.solver_margin_ori)

        if ik_js is None:
            rospy.logerr('No IK solution found')
            return None
        self.group.set_joint_value_target(np.array(ik_js))
        plan_goal = self.group.plan()
        return plan_goal

    def update_seed_state(self):
        panda_joints = rospy.wait_for_message('panda/joint_states', JointState)
        self.seed_state = list(panda_joints.position)

    def handle_plan_arm_trajectory(self, req):
        self.update_seed_state()
        plan = None
        plan = self.go_goal_trac_ik(req.palm_goal_pose_world)

        res = PlanArmTrajectoryResponse()
        if plan is None:
            res.success = False
            return res
        if len(plan.joint_trajectory.points) > 0:
            res.success = True
            res.trajectory = plan.joint_trajectory
        return res

    def handle_plan_arm_reset_trajectory(self, req):
        panda_joint_state = rospy.wait_for_message('panda/joint_states', JointState)
        diff = np.abs(self.home_joint_states - np.array(panda_joint_state.position))
        res = PlanResetTrajectoryResponse()
        if np.sum(diff) > 0.3:
            self.update_seed_state()
            plan = None
            plan = self.go_home()
        else:
            res.success = False
            return res

        if plan is None:
            rospy.loginfo("No plan for going home could be found.")
            res.success = False
            return res
        else:
            res.success = True
            res.trajectory = plan.joint_trajectory
        return res

    def create_plan_arm_trajectory_server(self):
        rospy.Service('plan_arm_trajectory', PlanArmTrajectory, self.handle_plan_arm_trajectory)
        rospy.loginfo('Service plan_arm_trajectory:')
        rospy.loginfo('Reference frame: %s' % self.group.get_planning_frame())
        rospy.loginfo('End-effector frame: %s' % self.group.get_end_effector_link())
        rospy.loginfo('Robot Groups: %s' % self.robot.get_group_names())
        rospy.loginfo('Ready to plan for given palm goal poses.')

    def create_plan_arm_reset_trajectory(self):
        rospy.Service('plan_arm_reset_trajectory', PlanResetTrajectory,
                      self.handle_plan_arm_reset_trajectory)
        rospy.loginfo('Service plan_arm_trajectory:')
        rospy.loginfo('Ready to plan reset trajectory')


if __name__ == '__main__':
    planner = CartesianPoseMoveitPlanner()
    planner.create_plan_arm_trajectory_server()
    planner.create_plan_arm_reset_trajectory()
    rospy.spin()