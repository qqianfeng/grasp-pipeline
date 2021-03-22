#!/usr/bin/env python
import rospy
import tf
import numpy as np
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from grasp_pipeline.srv import *


class BroadcastTf:
    '''
    The class to 1. create ROS services to update the grasp palm tf;
    2. broadcast the updated grasp palm tf; 3. update and broadcast the object tf.
    '''
    def __init__(self):
        rospy.init_node('grasp_palm_tf_br')
        # Grasp pose in blensor camera frame.
        self.palm_pose = None
        self.object_pose_world = None
        self.object_mesh_frame_pose = None
        self.object_mesh_frame_data_gen = None
        self.grasp_pose_list = None
        self.tf_br = tf.TransformBroadcaster()

    def broadcast_tf(self):
        '''
        Broadcast the grasp palm tf.
        '''
        if self.palm_pose is not None:
            self.tf_br.sendTransform(
                (self.palm_pose.pose.position.x, self.palm_pose.pose.position.y,
                 self.palm_pose.pose.position.z),
                (self.palm_pose.pose.orientation.x, self.palm_pose.pose.orientation.y,
                 self.palm_pose.pose.orientation.z, self.palm_pose.pose.orientation.w),
                rospy.Time.now(), 'grasp_palm_pose', self.palm_pose.header.frame_id)

        if self.object_pose_world is not None:
            self.tf_br.sendTransform(
                (self.object_pose_world.pose.position.x, self.object_pose_world.pose.position.y,
                 self.object_pose_world.pose.position.z),
                (self.object_pose_world.pose.orientation.x,
                 self.object_pose_world.pose.orientation.y,
                 self.object_pose_world.pose.orientation.z,
                 self.object_pose_world.pose.orientation.w), rospy.Time.now(),
                'object_pose_aligned', self.object_pose_world.header.frame_id)

        if self.object_mesh_frame_pose is not None:
            self.tf_br.sendTransform((self.object_mesh_frame_pose.pose.position.x,
                                      self.object_mesh_frame_pose.pose.position.y,
                                      self.object_mesh_frame_pose.pose.position.z),
                                     (self.object_mesh_frame_pose.pose.orientation.x,
                                      self.object_mesh_frame_pose.pose.orientation.y,
                                      self.object_mesh_frame_pose.pose.orientation.z,
                                      self.object_mesh_frame_pose.pose.orientation.w),
                                     rospy.Time.now(), 'object_mesh_frame',
                                     self.object_mesh_frame_pose.header.frame_id)

        if self.object_mesh_frame_data_gen is not None:
            self.tf_br.sendTransform((self.object_mesh_frame_data_gen.pose.position.x,
                                      self.object_mesh_frame_data_gen.pose.position.y,
                                      self.object_mesh_frame_data_gen.pose.position.z),
                                     (self.object_mesh_frame_data_gen.pose.orientation.x,
                                      self.object_mesh_frame_data_gen.pose.orientation.y,
                                      self.object_mesh_frame_data_gen.pose.orientation.z,
                                      self.object_mesh_frame_data_gen.pose.orientation.w),
                                     rospy.Time.now(), 'object_mesh_frame_data_gen',
                                     self.object_mesh_frame_data_gen.header.frame_id)

        if self.grasp_pose_list is not None:
            for i, pose in enumerate(self.grasp_pose_list):
                self.tf_br.sendTransform(
                    (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z),
                    (pose.pose.orientation.x,
                     pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w),
                    rospy.Time.now(), 'grasp_pose_' + str(i).zfill(3), pose.header.frame_id)

    def handle_update_palm_pose(self, req):
        '''
        Handler to update the palm pose tf.
        '''
        self.palm_pose = req.palm_pose
        return UpdatePalmPoseResponse(success=True)

    def handle_update_object_pose(self, req):
        '''
        Handler to update the object pose tf.
        '''
        self.object_pose_world = req.object_pose_world
        return UpdateObjectPoseResponse(success=True)

    def handle_update_object_mesh_frame_pose(self, req):
        self.object_mesh_frame_pose = req.object_pose_world
        return UpdateObjectPoseResponse(success=True)

    def handle_update_object_mesh_frame_data_gen(self, req):
        self.object_mesh_frame_data_gen = req.object_pose_world
        return UpdateObjectPoseResponse(success=True)

    def handle_visualize_grasp_pose_list(self, req):
        """ Visualize all grasp poses.
        """
        self.grasp_pose_list = req.grasp_pose_list
        return VisualizeGraspPoseListResponse(success=True)

    def create_update_palm_pose_server(self):
        '''
        Create the ROS server to update the palm tf.
        '''
        rospy.Service('update_grasp_palm_pose', UpdatePalmPose, self.handle_update_palm_pose)
        rospy.loginfo('Service update_grasp_palm_pose:')
        rospy.loginfo('Ready to update grasp palm pose:')

    def create_update_object_pose_server(self):
        '''
        Create the ROS server to update the object tf.
        '''
        rospy.Service('update_grasp_object_pose', UpdateObjectPose, self.handle_update_object_pose)
        rospy.loginfo('Service update_grasp_object_pose:')
        rospy.loginfo('Ready to update grasp object pose:')

    def create_update_object_mesh_frame_pose_server(self):
        '''
        Create the ROS server to update the object mesh tf.
        '''
        rospy.Service('update_object_mesh_frame_pose', UpdateObjectPose,
                      self.handle_update_object_mesh_frame_pose)
        rospy.loginfo('Service update_object_mesh_frame_pose:')
        rospy.loginfo('Ready to update_object_mesh_frame_pose:')

    def create_update_object_mesh_frame_data_gen_server(self):
        '''
        Create the ROS server to update the object mesh frame data gen tf.
        '''
        rospy.Service('update_object_mesh_frame_data_gen', UpdateObjectPose,
                      self.handle_update_object_mesh_frame_data_gen)
        rospy.loginfo('Service update_object_mesh_frame_data_gen:')
        rospy.loginfo('Ready to update_object_mesh_frame_data_gen:')

    def create_visualize_grasp_pose_list_server(self):
        rospy.Service('visualize_grasp_pose_list', VisualizeGraspPoseList,
                      self.handle_visualize_grasp_pose_list)
        rospy.loginfo('Service visualize_grasp_pose_list')
        rospy.loginfo('Ready to visualize_grasp_pose_list')


if __name__ == '__main__':
    btf = BroadcastTf()
    btf.create_update_palm_pose_server()
    btf.create_update_object_pose_server()
    btf.create_update_object_mesh_frame_pose_server()
    btf.create_update_object_mesh_frame_data_gen_server()
    btf.create_visualize_grasp_pose_list_server()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        btf.broadcast_tf()
        rate.sleep()
