#!/usr/bin/env python
import rospy
import time
import tf
from grasp_pipeline.srv import *
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trac_ik_python.trac_ik import IK


class PalmGoalPosesFilter():
    def __init__(self):
        rospy.init_node('filter_palm_goal_poses_node')
        self.tf_broadcaster_palm_poses = tf.TransformBroadcaster()

        self.state_validity_service = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        self.robot_state = RobotState()
        self.robot_state.joint_state.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5',
            'panda_joint6', 'panda_joint7'
        ]
        self.home_joint_states = [0, 0, 0, -1, 0, 1.9884, -1.57]

        self.ik_solver = IK(
            "panda_link0",
            "palm_link_hithand",
            timeout=0.01,
            epsilon=1e-4,
            solve_type="Manipulation1")
        self.seed_state = self.home_joint_states
        self.solver_margin_pos = 0.005
        self.solver_margin_ori = 0.01

        self.service_is_called = False

    def broadcast_palm_poses(self):
        if self.service_is_called:
            # Publish the palm goal tf
            for i, palm_pose_world in enumerate(self.pub_poses):
                self.tf_broadcaster_palm_poses.sendTransform(
                    (palm_pose_world.pose.position.x, palm_pose_world.pose.position.y,
                     palm_pose_world.pose.position.z),
                    (palm_pose_world.pose.orientation.x, palm_pose_world.pose.orientation.y,
                     palm_pose_world.pose.orientation.z, palm_pose_world.pose.orientation.w),
                    rospy.Time.now(), 'filt_' + str(i), palm_pose_world.header.frame_id)

    def get_ik_for_palm_pose(self, pose):
        ik_js = self.ik_solver.get_ik(
            self.seed_state, pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
            pose.pose.orientation.w, self.solver_margin_pos, self.solver_margin_pos,
            self.solver_margin_pos, self.solver_margin_ori, self.solver_margin_ori,
            self.solver_margin_ori)

        return ik_js

    @staticmethod
    def check_if_is_part_of_hand_link(str):
        if str[-7:] == 'hithand' or str[:5] == 'Right':
            return True
        else:
            return False

    def check_pose_for_collision(self, ik_js):
        """It seems this checks the collision with environments and self collision, the feedback found online. But not verified so far."""
        gsvr = GetStateValidityRequest()
        self.robot_state.joint_state.position = ik_js
        gsvr.robot_state = self.robot_state
        gsvr.group_name = 'panda_arm'
        result = self.state_validity_service(gsvr)
        # manually check if there is any collision for the hand
        # ignore the collision from the arm itself
        for contact in result.contacts:
            if self.check_if_is_part_of_hand_link(contact.contact_body_1) or self.check_if_is_part_of_hand_link(contact.contact_body_2):
                return True

        return False

    def handle_filter_palm_goal_poses(self, req):
        """ Receives a list of all palm goal poses (grasp hypotheses) and returns a list of idxs of unfeasible grasps, either because
        no IK solution could be found or the pose is in collision
        """
        # TODO: it hangs here once, after print "filtering palm goal poses", it hangs.
        print('Filtering palm goal poses.')
        prune_idxs = []
        no_ik_idxs = []
        collision_idxs = []
        self.pub_poses = []
        goal_poses = req.palm_goal_poses_world
        print("will iterate through: ", len(goal_poses), "poses")
        for i, pose in enumerate(goal_poses):
            ik_js = self.get_ik_for_palm_pose(pose)
            if ik_js is not None:
                result = self.check_pose_for_collision(ik_js)
                if result:
                    prune_idxs.append(i)
                    collision_idxs.append(i)
                else:
                    self.pub_poses.append(pose)
            else:
                prune_idxs.append(i)
                no_ik_idxs.append(i)
        print(str(len(no_ik_idxs)) + ' poses have no ik solution, ' + str(len(collision_idxs)) +
              ' poses would cause collision.')
        print(str(len(prune_idxs)) + ' out of ' + str(len(goal_poses)) + ' points filtered out.')
        self.service_is_called = True
        # Return the filtered poses which are not in collision
        res = FilterPalmPosesResponse()
        res.prune_idxs = prune_idxs
        res.no_ik_idxs = no_ik_idxs
        res.collision_idxs = collision_idxs
        return res

    def create_filter_palm_goal_poses_server(self):
        rospy.Service('filter_palm_goal_poses', FilterPalmPoses,
                      self.handle_filter_palm_goal_poses)
        rospy.loginfo('Service filter_palm_goal_poses is ready.')


if __name__ == '__main__':
    pgf = PalmGoalPosesFilter()
    pgf.create_filter_palm_goal_poses_server()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pgf.broadcast_palm_poses()
        rate.sleep()