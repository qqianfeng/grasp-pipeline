#!/usr/bin/env python
import rospy
import tf
import tf.transformations as tft
from grasp_pipeline.srv import *
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
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
        self.move_group = moveit_commander.MoveGroupCommander('panda_arm')

        self.move_group.set_planner_id('RRTConnectkConfigDefault')
        self.move_group.set_planning_time(10)
        self.move_group.set_num_planning_attempts(3)

        self.home_joint_states = np.array([0, 0, 0, -1, 0, 1.9884, -1.57])

        self.ik_solver = IK("world", "palm_link_hithand", timeout=0.5,
                            epsilon=1e-4)  #panda_link0 was world before
        self.seed_state = [0.0] * self.ik_solver.number_of_joints
        self.solver_margin_pos = 0.005
        self.solver_margin_ori = 0.01

        self.tf_listener = tf.TransformListener()

    def go_home(self):
        print 'go home'
        self.move_group.clear_pose_targets()
        self.move_group.set_joint_value_target(self.home_joint_states)
        plan_home = self.move_group.plan()
        return plan_home

    def go_goal_trac_ik(self, pose):
        print 'go goal trac ik'
        self.move_group.clear_pose_targets()
        ik_js = self.ik_solver.get_ik(
            self.seed_state, pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
            pose.pose.orientation.w, self.solver_margin_pos, self.solver_margin_pos,
            self.solver_margin_pos, self.solver_margin_ori, self.solver_margin_ori,
            self.solver_margin_ori)

        if ik_js is None:
            rospy.logerr('No IK solution found')
            return None
        self.move_group.set_joint_value_target(np.array(ik_js))
        plan_goal = self.move_group.plan()
        return plan_goal

    def update_seed_state(self):
        panda_joints = rospy.wait_for_message('panda/joint_states', JointState)
        self.seed_state = list(panda_joints.position)

    def get_ee_pose(self):
        (trans, quat) = self.tf_listener.lookupTransform('world',
                                                         self.move_group.get_end_effector_link(),
                                                         rospy.Time())
        ee_pose = Pose()
        ee_pose.position.x = trans[0]
        ee_pose.position.y = trans[1]
        ee_pose.position.z = trans[2]
        ee_pose.orientation.x = quat[0]
        ee_pose.orientation.y = quat[1]
        ee_pose.orientation.z = quat[2]
        ee_pose.orientation.w = quat[3]
        return ee_pose

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

    def handle_plan_reset_trajectory(self, req):
        self.update_seed_state()
        diff = np.abs(self.home_joint_states - np.array(self.seed_state))
        res = PlanResetTrajectoryResponse()
        if np.sum(diff) > 0.3:
            plan = None
            plan = self.go_home()
        else:
            res.success = False
            return res

        if plan is None:
            rospy.loginfo("No plan for going home could be found.")
            res.success = False
            return res
        if len(plan.joint_trajectory.points) == 0:
            rospy.loginfo("Panda is already in home position, no need to reset.")
            res.success = False
        else:
            res.success = True
            res.trajectory = plan.joint_trajectory
        return res

    def handle_plan_cartesian_path_trajectory(self, req):
        plan = None

        # Add 3 waypoints: current pose, approach pose and goal pose
        waypoints = []
        wpose = self.get_ee_pose()
        waypoints.append(wpose)
        waypoints.append(req.palm_approach_pose_world.pose)
        waypoints.append(req.palm_goal_pose_world.pose)

        # Plan path through these waypoints
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)

        res = PlanCartesianPathTrajectoryResponse()
        if plan is None:
            res.success = False
            return res
        if len(plan.joint_trajectory.points) > 0:
            res.success = True
            res.trajectory = plan.joint_trajectory
        return res

    def create_plan_arm_trajectory_server(self):
        rospy.Service('plan_arm_trajectory', PlanArmTrajectory, self.handle_plan_arm_trajectory)
        rospy.loginfo('Service plan_arm_trajectory:')
        rospy.loginfo('Reference frame: %s' % self.move_group.get_planning_frame())
        rospy.loginfo('End-effector frame: %s' % self.move_group.get_end_effector_link())
        rospy.loginfo('Ready to plan for given palm goal poses.')

    def create_plan_arm_reset_trajectory(self):
        rospy.Service('plan_reset_trajectory', PlanResetTrajectory,
                      self.handle_plan_reset_trajectory)
        rospy.loginfo('Service plan_reset_trajectory:')
        rospy.loginfo('Ready to plan reset trajectory')

    def create_plan_cartesian_path_trajectory(self):
        rospy.Service('plan_cartesian_path_trajectory', PlanCartesianPathTrajectory,
                      self.handle_plan_cartesian_path_trajectory)
        rospy.loginfo('Service plan_cartesian_path_trajectory:')
        rospy.loginfo('Ready to plan cartesian_path trajectory')


if __name__ == '__main__':
    planner = CartesianPoseMoveitPlanner()
    planner.create_plan_arm_trajectory_server()
    planner.create_plan_arm_reset_trajectory()
    planner.create_plan_cartesian_path_trajectory()
    rospy.spin()
