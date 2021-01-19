#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState
import numpy as np
from visualization_msgs.msg import InteractiveMarkerUpdate, InteractiveMarkerPose
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBoolRequest, SetBool


class TestGraspController():
    def __init__(self):
        rospy.init_node('test_grasp_controller')
        self.robot_cartesian_goal_pose_pandaj8 = None
        self.joint_trajectory_to_goal = None
        self.robot_cartesian_goal_pose = None
        self.setup_lift_and_grasp_poses()

    def setup_lift_and_grasp_poses(self):
        self.lift_pose = InteractiveMarkerPose()
        self.lift_pose.pose.position.x = 0.6476439232691327
        self.lift_pose.pose.position.y = -0.009837140154184118
        self.lift_pose.pose.position.z = 0.507262644093148

        self.lift_pose.pose.orientation.x = 0.8476802233161043
        self.lift_pose.pose.orientation.y = 0.5240429764425666
        self.lift_pose.pose.orientation.z = -0.03652414525949581
        self.lift_pose.pose.orientation.w = 0.07404852904034345

        self.grasp_pose = InteractiveMarkerPose()
        self.grasp_pose.pose.position.x = 0.6615187681665692
        self.grasp_pose.pose.position.y = -0.04421595156309393
        self.grasp_pose.pose.position.z = 0.3212456481334106
        self.grasp_pose.pose.orientation.x = 0.7901923091273372
        self.grasp_pose.pose.orientation.y = 0.607271084365329
        self.grasp_pose.pose.orientation.z = -0.04383754600659636
        self.grasp_pose.pose.orientation.w = 0.06997295370272645

    def transform_pose_to_palm_link(self, pose, from_link='panda_link8'):
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        rospy.sleep(0.5)
        transform = tfBuffer.lookup_transform(from_link, 'palm_link_hithand', rospy.Time())
        trans_tf_mat = tft.translation_matrix([
            transform.transform.translation.x, transform.transform.translation.y,
            transform.transform.translation.z
        ])
        rot_tf_mat = tft.quaternion_matrix([
            transform.transform.rotation.x, transform.transform.rotation.y,
            transform.transform.rotation.z, transform.transform.rotation.w
        ])
        palm_T_link8 = np.dot(trans_tf_mat, rot_tf_mat)

        trans_pose_mat = tft.translation_matrix([
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
        ])
        rot_pose_mat = tft.quaternion_matrix([
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
            pose.pose.orientation.w
        ])
        link8_T_world = np.dot(trans_pose_mat, rot_pose_mat)

        palm_T_world = np.dot(link8_T_world, palm_T_link8)

        palm_pose_stamped = PoseStamped()
        palm_pose_stamped.header.frame_id = 'world'
        palm_pose_stamped.pose.position.x = palm_T_world[0, 3]
        palm_pose_stamped.pose.position.y = palm_T_world[1, 3]
        palm_pose_stamped.pose.position.z = palm_T_world[2, 3]

        quat = tft.quaternion_from_matrix(palm_T_world)

        palm_pose_stamped.pose.orientation.x = quat[0]
        palm_pose_stamped.pose.orientation.y = quat[1]
        palm_pose_stamped.pose.orientation.z = quat[2]
        palm_pose_stamped.pose.orientation.w = quat[3]

        return palm_pose_stamped

    def get_goal_pose_from_marker(self):
        while True:
            marker_pose = rospy.wait_for_message(
                '/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update',
                InteractiveMarkerUpdate, 10)
            if len(marker_pose.poses) != 0:
                break
        palm_T_world = self.transform_pose_to_palm_link(marker_pose.poses[0])
        self.robot_cartesian_goal_pose = palm_T_world

    def plan_joint_trajectory_to_goal(self):
        arm_moveit_cartesian_pose_planner = rospy.ServiceProxy('arm_moveit_cartesian_pose_planner',
                                                               PalmGoalPoseWorld)
        req = PalmGoalPoseWorldRequest()
        req.palm_goal_pose_world = self.robot_cartesian_goal_pose
        res = arm_moveit_cartesian_pose_planner(req)
        self.joint_trajectory_to_goal = res.plan_traj
        return res.success

    def execute_joint_trajectory_to_goal(self):
        execute_joint_trajectory = rospy.ServiceProxy('execute_joint_trajectory',
                                                      ExecuteJointTrajectory)
        req = ExecuteJointTrajectoryRequest()
        req.smoothen_trajectory = True
        req.joint_trajectory = self.joint_trajectory_to_goal
        res = execute_joint_trajectory(req)
        return res.success

    def move_to_marker_position(self):
        self.get_goal_pose_from_marker()
        result = self.plan_joint_trajectory_to_goal()
        # try replanning 5 times
        if result == False:
            for i in range(5):
                result = self.plan_joint_trajectory_to_goal()
                if result:
                    break
        self.execute_joint_trajectory_to_goal()

    def move_to_lift_position(self):
        palm_T_world = self.transform_pose_to_palm_link(self.lift_pose)
        self.robot_cartesian_goal_pose = palm_T_world
        result = self.plan_joint_trajectory_to_goal()
        # try replanning 5 times
        if result == False:
            for i in range(5):
                result = self.plan_joint_trajectory_to_goal()
                if result:
                    break
        self.execute_joint_trajectory_to_goal()

    def move_to_grasp_position(self):
        palm_T_world = self.transform_pose_to_palm_link(self.grasp_pose)
        self.robot_cartesian_goal_pose = palm_T_world
        result = self.plan_joint_trajectory_to_goal()
        # try replanning 5 times
        if result == False:
            for i in range(5):
                result = self.plan_joint_trajectory_to_goal()
                if result:
                    break
        self.execute_joint_trajectory_to_goal()

    def spawn_object(self):
        update_gazebo_object = rospy.ServiceProxy('update_gazebo_object', UpdateObjectGazebo)
        pose = [0, 0, 0, 0.7, 0, 0]
        update_gazebo_object('mustard_bottle', pose, '006_mustard_bottle', 'sdf', 'ycb')

    def grasp_object(self):
        grasp_control_hithand = rospy.ServiceProxy('/grasp_control_hithand', GraspControl)
        req = GraspControlRequest()
        res = grasp_control_hithand(req)

    def reset_hand(self):
        reset_hithand_joints = rospy.ServiceProxy('/reset_hithand_joints', SetBool)
        req = SetBoolRequest()
        res = reset_hithand_joints(req)


if __name__ == '__main__':
    t = TestGraspController()
    raw_input('Press enter to continue: ')
    t.reset_hand()
    rospy.sleep(1)
    t.spawn_object()
    raw_input('Press enter to continue: ')
    # #t.move_to_marker_position()
    t.move_to_grasp_position()
    raw_input('Press enter to continue: ')
    t.grasp_object()
    raw_input('Press enter to continue: ')
    t.move_to_lift_position()
