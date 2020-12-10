#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from std_msgs.msg import Float32MultiArray
import open3d as o3d
from tf import TransformListener
import numpy as np
import copy
from geometry_msgs.msg import PointStamped

object_bounding_box_corner_points = np.array(
    [[0.73576546, -0.05682119, 0.01387128],
     [0.83193374, -0.05682119, 0.01387128],
     [0.73576546, -0.01262708, 0.01387128],
     [0.73576546, -0.05682119, 0.19168445],
     [0.83193374, -0.01262708, 0.19168445],
     [0.73576546, -0.01262708, 0.19168445],
     [0.83193374, -0.05682119, 0.19168445],
     [0.83193374, -0.01262708, 0.01387128]])
bbp1, bbp2, bbp3, bbp4 = object_bounding_box_corner_points[
    0, :], object_bounding_box_corner_points[
        1, :], object_bounding_box_corner_points[
            2, :], object_bounding_box_corner_points[3, :],
bbp5, bbp6, bbp7, bbp8 = object_bounding_box_corner_points[
    4, :], object_bounding_box_corner_points[
        5, :], object_bounding_box_corner_points[
            6, :], object_bounding_box_corner_points[7, :],


class BoundingBoxFace():
    """Simple class to store properties of a bounding box face.
    """
    def __init__(self,
                 center,
                 orient_a,
                 orient_b,
                 height,
                 width,
                 is_top=False):
        self.center = center
        self.orient_a = orient_a
        self.orient_b = orient_b
        self.height = height
        self.width = width
        self.is_top = is_top


class GenerateHithandPreshape():
    """ Generates Preshapes for the Hithand via sampling or geometric considerations.

    It stores the object_size, bounding_box_corner_points and objct_pcd in instance variables
    Returned should be the hithand joint angles.
    """
    def __init__(self):
        rospy.init_node("generate_hithand_preshape_node")
        self.min_object_height = 0.05  # object must be at least 5cm tall in order for side grasps to have a chance
        self.object_size = None
        self.object_bounding_box_corner_points = None
        self.segmented_object_pcd = None
        self.bbp1, self.bbp2, self.bbp3, self.bbp4, self.bbp5, self.bbp6, self.bbp7, self.bbp8 = 8 * [
            None
        ]
        self.palm_goal_pose_world = [
        ]  # This stores the 6D palm pose in the world

    def update_object_information(self):
        """ Update instance variables related to the object of interest

        This is intended to 1.) receive a single message from the segmented_object topics and store them in instance attributes and 
        2.) read the segmented object point cloud from disk
        """
        self.object_size = rospy.wait_for_message('/segmented_object_size',
                                                  Float32MultiArray).data
        self.object_bounding_box_corner_points = np.array(
            rospy.wait_for_message(
                '/segmented_object_bounding_box_corner_points',
                Float32MultiArray).data)
        self.bbp1, self.bbp2, self.bbp3, self.bbp4 = self.object_bounding_box_corner_points[
            0, :], self.object_bounding_box_corner_points[
                1, :], self.object_bounding_box_corner_points[
                    2, :], self.object_bounding_box_corner_points[3, :],
        self.bbp5, self.bbp6, self.bbp7, self.bbp8 = self.object_bounding_box_corner_points[
            4, :], self.object_bounding_box_corner_points[
                5, :], self.object_bounding_box_corner_points[
                    6, :], self.object_bounding_box_corner_points[7, :],
        self.segmented_object_pcd = o3d.io.read_point_cloud(
            '/home/vm/object.pcd')

    def find_palm_pose_from_bounding_box_centers(self,
                                                 bounding_box_center_points,
                                                 bounding_box_faces):
        pass

    def get_axis_aligned_bounding_box_faces_and_center(self):
        """ Get the center points of 3 axis aligned bounding box faces, 1 top and 2 closest to camera

        Black, R, G, B --> Bounding Box Corner Points 1,2,3,4. Note: Assumes axis-aligned bounding box, needs to be adapted for oriented bounding boxes. 
        """
        # The 1. and 5. point of bounding_box_corner points are cross-diagonal
        # Also the bounding box axis are aligned to the world frame
        x_axis, y_axis, z_axis = np.array([1, 0, 0]), np.array(
            [0, 1, 0]), np.array([0, 0, 1])
        size_x, size_y, size_z = self.object_size[0], self.object_size[
            1], self.object_size[2]
        #bounding_box_center = 0.5 * (self.bbp1 + self.bbp5)
        faces_world_frame = [
            BoundingBoxFace(center=0.5 * (self.bbp1 + self.bbp6),
                            orient_a=y_axis,
                            orient_b=z_axis,
                            height=size_y,
                            width=size_z),
            BoundingBoxFace(center=0.5 * (self.bbp1 + self.bbp7),
                            orient_a=x_axis,
                            orient_b=z_axis,
                            height=size_x,
                            width=size_z),
            BoundingBoxFace(center=0.5 * (self.bbp1 + self.bbp8),
                            orient_a=x_axis,
                            orient_b=y_axis,
                            height=size_x,
                            width=size_y),
            BoundingBoxFace(center=0.5 * (self.bbp5 + self.bbp2),
                            orient_a=y_axis,
                            orient_b=z_axis,
                            height=size_y,
                            width=size_z),
            BoundingBoxFace(center=0.5 * (self.bbp5 + self.bbp3),
                            orient_a=x_axis,
                            orient_b=y_axis,
                            height=size_x,
                            width=size_y),
            BoundingBoxFace(center=0.5 * (self.bbp5 + self.bbp4),
                            orient_a=x_axis,
                            orient_b=z_axis,
                            height=size_x,
                            width=size_z)
        ]
        # find the top face
        faces_world_frame = sorted(faces_world_frame,
                                   key=lambda x: x.center[2])
        faces_world_frame[-1].is_top = True
        # Delete the bottom face
        del faces_world_frame[0]
        # Sort along the x axis and delete the face furthes away (robot can't comfortably reach it)
        faces_world_frame = sorted(faces_world_frame,
                                   key=lambda x: x.center[0])
        del faces_world_frame[-1]
        # Sort along y axis and delete the face furthest away (no normals in this area)
        faces_world_frame = sorted(faces_world_frame,
                                   key=lambda x: x.center[1])
        del faces_world_frame[-1]
        # Create stamped poses for faces center position
        face_centers_world_frame = []
        center_stamped_world = PointStamped()
        center_stamped_world.header.frame_id = 'world'
        for _, face in enumerate(faces_world_frame):
            center_stamped_world.point.x = face.center[0]
            center_stamped_world.point.y = face.center[1]
            center_stamped_world.point.z = face.center[2]
            face_centers_world_frame.append(
                copy.deepcopy(center_stamped_world))

        #They also publish the points, not sure why needed: publish_points(self.grasp_pose_pub, face_centers_world_frame)

        # If the object is too short, only select top grasps.
        rospy.loginfo('##########################')
        rospy.loginfo('Obj_height: %s' % size_z)
        if size_z < self.min_object_height:
            rospy.loginfo('Object is short, only use top grasps!')
            return [face_centers_world_frame[0]], [faces_world_frame[0]]

        return face_centers_world_frame, faces_world_frame

    def sample_grasp_preshape(self, req):
        """ Grasp preshape service callback for sampling Hithand grasp preshapes.
        """
        res = GraspPreshapeResponse()
        # Compute bounding box faces
        bounding_box_center_points, bounding_box_faces = self.get_axis_aligned_bounding_box_faces_and_center(
        )
        self.palm_goal_pose_world = []
        for i in xrange(len(bounding_box_faces)):
            # Get the desired palm pose given the point from the bounding box
            palm_pose_world = self.find_palm_pose_from_bounding_box_centers(
                bounding_box_center_points, bounding_box_faces)
            self.palm_goal_pose_world.append(palm_pose_world)

            self.set_palm_rand_pose_limits(palm_pose_world)

    def generate_grasp_preshape(self, req):
        res = GraspPreshapeResponse()
        return res

    def handle_generate_hithand_preshape(self, req):
        """ Handler for the advertised service. Decides whether the hand preshape should be sampled or generated
        """
        self.update_object_information(
        )  # Get new information on segmented object from rostopics/disk and store in instance attributes
        if req.sample:
            return self.sample_grasp_preshape(req)
        else:
            return self.generate_grasp_preshape(req)

    def create_hithand_preshape_server(self):
        rospy.Service('generate_hithand_preshape_service', GraspPreshape,
                      self.handle_generate_hithand_preshape)
        rospy.loginfo('Service generate_hithand_preshape:')
        rospy.loginfo('Ready to generate the grasp preshape.')


if __name__ == '__main__':
    ghp = GenerateHithandPreshape()
    #ghp.create_hithand_preshape_server()
    ghp.get_axis_aligned_bounding_box_faces_and_center()

    rospy.spin()