#!/usr/bin/env python
import copy
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

        self.ik_solver = IK("panda_link0",
                            "palm_link_hithand",
                            timeout=0.01,
                            epsilon=1e-4,
                            solve_type="Manipulation1")  #panda_link0 was world before
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
            rospy.logdebug('No IK solution found')
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
        """It's a move joint command.

        """
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
        waypoints.append(self.get_ee_pose())

        waypoints.append(copy.deepcopy(req.palm_approach_pose_world.pose))
        waypoints.append(copy.deepcopy(req.palm_goal_pose_world.pose))

        # Plan path through these waypoints
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        rospy.loginfo('fraction: %f' % fraction)
        res = PlanCartesianPathTrajectoryResponse()
        if plan is None:
            res.success = False
            return res
        if len(plan.joint_trajectory.points) > 0:
            res.success = True
            res.trajectory = plan.joint_trajectory
            rospy.loginfo('plan.joint_trajectory is %s' % str(plan.joint_trajectory.points))
            res.fraction = fraction
        return res

    def handle_check_cartesian_pose_distance(self, req):
        current_pose = self.get_ee_pose()
        required_pose = req.required_pose.pose
        delta_x = abs(current_pose.position.x - required_pose.position.x)
        delta_y = abs(current_pose.position.y - required_pose.position.y)
        delta_z = abs(current_pose.position.z - required_pose.position.z)
        distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        rospy.logdebug("delta_x, %f" % (current_pose.position.x - required_pose.position.x))
        rospy.logdebug("delta_y, %f" % (current_pose.position.y - required_pose.position.y))
        rospy.logdebug("delta_z, %f" % (current_pose.position.z - required_pose.position.z))
        res = CheckCartesianPoseDistanceResponse()
        res.distance = distance
        return res

    def create_check_cartesian_pose_distance(self):
        rospy.Service('check_cartesian_pose_distance', CheckCartesianPoseDistance,
                      self.handle_check_cartesian_pose_distance)
        rospy.loginfo('Service check_cartesian_pose_distance:')
        rospy.loginfo('Ready to calculate catesian goal pose distance.')

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
    planner.create_check_cartesian_pose_distance()

    DEBUG = False
    if DEBUG:

        ee_pose = PoseStamped()
        ee_pose.pose.position.x = 0.48074272179180777
        ee_pose.pose.position.y = 0.1280189956862769
        ee_pose.pose.position.z = 0.06839975362456746
        ee_pose.pose.orientation.x = 0.5478170055099875
        ee_pose.pose.orientation.y = 0.013352065752940073
        ee_pose.pose.orientation.z = 0.07366870805281238
        ee_pose.pose.orientation.w = 0.8332413555246935

        via_pose = PoseStamped()
        via_pose.pose.position.x = 0.48074272179180777
        via_pose.pose.position.y = 0.1280189956862769
        via_pose.pose.position.z = 0.06839975362456746
        via_pose.pose.orientation.x = 0.5478170055099875
        via_pose.pose.orientation.y = 0.013352065752940073
        via_pose.pose.orientation.z = 0.07366870805281238
        via_pose.pose.orientation.w = 0.8332413555246935
        plan_cartesian_path_trajectory = rospy.ServiceProxy('plan_cartesian_path_trajectory',
                                                            PlanCartesianPathTrajectory)
        req = PlanCartesianPathTrajectoryRequest()
        req.palm_approach_pose_world = via_pose
        req.palm_goal_pose_world = ee_pose
        res = plan_cartesian_path_trajectory(req)

    rospy.spin()
