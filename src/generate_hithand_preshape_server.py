#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from std_msgs.msg import Float32MultiArray
import open3d as o3d
from tf import TransformListener
import numpy as np
import copy
from geometry_msgs.msg import PointStamped, PoseStamped
import tf.transformations as tft

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
        self.center = np.array(center)
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
        # Init node
        rospy.init_node("generate_hithand_preshape_node")

        # Get parameters from the ROS parameter server
        self.min_object_height = rospy.get_param(
            '~min_object_height', 0.03
        )  # object must be at least 5cm tall in order for side grasps to have a chance
        self.palm_dist_to_top_face_mean = rospy.get_param(
            '~palm_dist_to_top_face_mean')
        self.palm_dist_to_side_face_mean = rospy.get_param(
            '~palm_dist_to_side_face_mean')
        self.palm_dist_normal_to_obj_var = rospy.get_param(
            '~palm_dist_normal_to_obj_var'
        )  # Determines how much variance is in the sampled palm distance in the normal direction
        self.palm_position_3D_sample_var = rospy.get_param(
            '~palm_position_3D_sample_var'
        )  # Some extra noise on the samples 3D palm position
        self.wrist_roll_orientation_var = rospy.get_param(
            '~wrist_roll_orientation_var'
        )  # Some extra noise on the samples 3D palm position
        self.use_bb_orient_to_determine_wrist_roll = True
        # Initialize object related instance variables
        self.object_size = None
        self.bounding_box_center = None
        self.object_bounding_box_corner_points = None
        self.bbp1, self.bbp2, self.bbp3, self.bbp4, self.bbp5, self.bbp6, self.bbp7, self.bbp8 = 8 * [
            None
        ]
        self.segmented_object_points = None
        self.segmented_object_normals = None

        self.palm_goal_pose_world = [
        ]  # This stores the 6D palm pose in the world

        self.palm_pose_lower_limit = None
        self.palm_pose_upper_limit = None

        self.num_samples_per_preshape = 50

    def update_object_information(self):
        """ Update instance variables related to the object of interest

        This is intended to 1.) receive a single message from the segmented_object topics and store them in instance attributes and 
        2.) read the segmented object point cloud from disk
        """
        # Size
        self.object_size = rospy.wait_for_message('/segmented_object_size',
                                                  Float32MultiArray).data
        # Bounding box corner points and center
        # The 1. and 5. point of bounding_box_corner points are cross-diagonal
        self.object_bounding_box_corner_points = np.array(
            rospy.wait_for_message(
                '/segmented_object_bounding_box_corner_points',
                Float32MultiArray).data)
        self.bbp1 = self.object_bounding_box_corner_points[0, :]
        self.bbp2 = self.object_bounding_box_corner_points[1, :]
        self.bbp3 = self.object_bounding_box_corner_points[2, :]
        self.bbp4 = self.object_bounding_box_corner_points[3, :]
        self.bbp5 = self.object_bounding_box_corner_points[4, :]
        self.bbp6 = self.object_bounding_box_corner_points[5, :]
        self.bbp7 = self.object_bounding_box_corner_points[6, :]
        self.bbp8 = self.object_bounding_box_corner_points[7, :]
        self.bounding_box_center = np.array(0.5 * (self.bbp1 + self.bbp5))

        # Object pcd, store points and normals
        segmented_object_pcd = o3d.io.read_point_cloud('/home/vm/object.pcd')
        segmented_object_pcd.normalize_normals()  # normalize the normals
        self.segmented_object_points = np.asarray(
            segmented_object_pcd.points)  # Nx3 shape
        self.segmented_object_normals = np.asarray(
            segmented_object_pcd.normals)

    def setup_joint_angle_limits(self):
        '''
        Initializes a number of constants determing the joint limits for allegro
        TODO: Automate this by using a URDF file and allow hand to be specified at launch
        '''
        self.index_joint_0_lower = -0.59
        self.index_joint_0_upper = 0.57
        self.middle_joint_0_lower = -0.59
        self.middle_joint_0_upper = 0.57
        self.ring_joint_0_lower = -0.59
        self.ring_joint_0_upper = 0.57

        self.index_joint_1_lower = -0.296
        self.index_joint_1_upper = 0.71
        self.middle_joint_1_lower = -0.296
        self.middle_joint_1_upper = 0.71
        self.ring_joint_1_lower = -0.296
        self.ring_joint_1_upper = 0.71

        self.thumb_joint_0_lower = 0.363
        self.thumb_joint_0_upper = 1.55
        self.thumb_joint_1_lower = -0.205
        self.thumb_joint_1_upper = 1.263

        self.index_joint_0_middle = (self.index_joint_0_lower +
                                     self.index_joint_0_upper) * 0.5
        self.middle_joint_0_middle = (self.middle_joint_0_lower +
                                      self.middle_joint_0_upper) * 0.5
        self.ring_joint_0_middle = (self.ring_joint_0_lower +
                                    self.ring_joint_0_upper) * 0.5
        self.index_joint_1_middle = (self.index_joint_1_lower +
                                     self.index_joint_1_upper) * 0.5
        self.middle_joint_1_middle = (self.middle_joint_1_lower +
                                      self.middle_joint_1_upper) * 0.5
        self.ring_joint_1_middle = (self.ring_joint_1_lower +
                                    self.ring_joint_1_upper) * 0.5
        self.thumb_joint_0_middle = (self.thumb_joint_0_lower +
                                     self.thumb_joint_0_upper) * 0.5
        self.thumb_joint_1_middle = (self.thumb_joint_1_lower +
                                     self.thumb_joint_1_upper) * 0.5

        self.index_joint_0_range = self.index_joint_0_upper - self.index_joint_0_lower
        self.middle_joint_0_range = self.middle_joint_0_upper - self.middle_joint_0_lower
        self.ring_joint_0_range = self.ring_joint_0_upper - self.ring_joint_0_lower
        self.index_joint_1_range = self.index_joint_1_upper - self.index_joint_1_lower
        self.middle_joint_1_range = self.middle_joint_1_upper - self.middle_joint_1_lower
        self.ring_joint_1_range = self.ring_joint_1_upper - self.ring_joint_1_lower
        self.thumb_joint_0_range = self.thumb_joint_0_upper - self.thumb_joint_0_lower
        self.thumb_joint_1_range = self.thumb_joint_1_upper - self.thumb_joint_1_lower

        self.first_joint_lower_limit = 0.25
        self.first_joint_upper_limit = 0.25
        self.second_joint_lower_limit = 0.5
        self.second_joint_upper_limit = 0.  #-0.1

        self.thumb_1st_joint_lower_limit = -0.5
        self.thumb_1st_joint_upper_limit = 0.5
        self.thumb_2nd_joint_lower_limit = 0.25
        self.thumb_2nd_joint_upper_limit = 0.25

        self.index_joint_0_sample_lower = self.index_joint_0_middle - self.first_joint_lower_limit * self.index_joint_0_range
        self.index_joint_0_sample_upper = self.index_joint_0_middle + self.first_joint_upper_limit * self.index_joint_0_range
        self.middle_joint_0_sample_lower = self.middle_joint_0_middle - self.first_joint_lower_limit * self.middle_joint_0_range
        self.middle_joint_0_sample_upper = self.middle_joint_0_middle + self.first_joint_upper_limit * self.middle_joint_0_range
        self.ring_joint_0_sample_lower = self.ring_joint_0_middle - self.first_joint_lower_limit * self.ring_joint_0_range
        self.ring_joint_0_sample_upper = self.ring_joint_0_middle + self.first_joint_upper_limit * self.ring_joint_0_range

        self.index_joint_1_sample_lower = self.index_joint_1_middle - self.second_joint_lower_limit * self.index_joint_1_range
        self.index_joint_1_sample_upper = self.index_joint_1_middle + self.second_joint_upper_limit * self.index_joint_1_range
        self.middle_joint_1_sample_lower = self.middle_joint_1_middle - self.second_joint_lower_limit * self.middle_joint_1_range
        self.middle_joint_1_sample_upper = self.middle_joint_1_middle + self.second_joint_upper_limit * self.middle_joint_1_range
        self.ring_joint_1_sample_lower = self.ring_joint_1_middle - self.second_joint_lower_limit * self.ring_joint_1_range
        self.ring_joint_1_sample_upper = self.ring_joint_1_middle + self.second_joint_upper_limit * self.ring_joint_1_range

        self.thumb_joint_0_sample_lower = self.thumb_joint_0_middle - self.thumb_1st_joint_lower_limit * self.thumb_joint_0_range
        self.thumb_joint_0_sample_upper = self.thumb_joint_0_middle + self.thumb_1st_joint_upper_limit * self.thumb_joint_0_range
        self.thumb_joint_1_sample_lower = self.thumb_joint_1_middle - self.thumb_2nd_joint_lower_limit * self.thumb_joint_1_range
        self.thumb_joint_1_sample_upper = self.thumb_joint_1_middle + self.thumb_2nd_joint_upper_limit * self.thumb_joint_1_range



    def sample_uniformly_around_preshape_palm_pose(self, frame_id):
        ''' Get a random palm pose by sampling around the preshape palm pose
        for sampling grasp detection. 

        The palm pose limit instance variables define the sampling range (middle between upper and lower limits is the previously computed palm pose)
        '''
        sample_palm_pose_array = np.random.uniform(self.palm_pose_lower_limit,
                                                   self.palm_pose_upper_limit)
        sample_palm_pose = PoseStamped()
        sample_palm_pose.header.frame_id = frame_id
        sample_palm_pose.pose.position.x, sample_palm_pose.pose.position.y, \
                sample_palm_pose.pose.position.z = sample_palm_pose_array[:3]

        palm_euler = sample_palm_pose_array[3:]
        palm_quaternion = tft.quaternion_from_euler(palm_euler[0],
                                                    palm_euler[1],
                                                    palm_euler[2])
        sample_palm_pose.pose.orientation = palm_quaternion

        return sample_palm_pose

    def set_palm_rand_pose_limits(self, palm_preshape_pose):
        """ Set the palm pose sample range for sampling grasp detection.
        """
        # Convert the pose into a format more amenable to subsequent task
        palm_preshape_euler = tft.euler_from_quaternion(
            (palm_preshape_pose.pose.orientation.x,
             palm_preshape_pose.pose.orientation.y,
             palm_preshape_pose.pose.orientation.z,
             palm_preshape_pose.pose.orientation.w))

        preshape_palm_pose_config = np.array([
            palm_preshape_pose.pose.position.x,
            palm_preshape_pose.pose.position.y,
            palm_preshape_pose.pose.position.z, palm_preshape_euler[0],
            palm_preshape_euler[1], palm_preshape_euler[2]
        ])

        # Add/ subtract these from the pose to get lower and upper limits
        pos_range = 0.05
        ort_range = 0.05 * np.pi
        upper_limit_range = np.array(
            [pos_range, pos_range, pos_range, ort_range, ort_range, ort_range])
        lower_limit_range = (-1) * upper_limit_range
        self.palm_pose_lower_limit = preshape_palm_pose_config + lower_limit_range
        self.palm_pose_upper_limit = preshape_palm_pose_config + upper_limit_range

    def sample_palm_orientation(self, object_point_normal,
                                bounding_box_orientation_vector):
        """ Sample the hand wrist roll uniformly. 

        For the palm frame, x: palm normal, y: thumb, z: middle finger
        """
        if self.use_bb_orient_to_determine_wrist_roll:
            y_rand = bounding_box_orientation_vector
        else:
            roll = np.random.uniform(-np.pi, np.pi)
            y_rand = np.array([0.0, np.sin(roll), np.cos(roll)])

        # Project orientation vector into the tangent space of the normal x tp get the y component, then determine z from the cross-product of the two
        x_axis = -object_point_normal
        y_onto_x = y_rand.dot(x_axis) * x_axis
        y_axis = (y_rand - y_onto_x) / np.linalg.norm(y_rand - y_onto_x)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        rot_matrix = np.matrix([x_axis, y_axis, z_axis]).T

        # Compute quaternion from rpy
        trans_matrix = np.matrix(np.zeros((4, 4)))
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[3, 3] = 1.
        quaternion = tft.quaternion_from_matrix(trans_matrix)
        rospy.loginfo('Grasp orientation = ' + str(quaternion))
        return quaternion

    def find_palm_pose_from_bounding_box_centers(self, bounding_box_face):
        """ Finds the object point cloud's point closest to the center of a given bounding box face
        and generates a Hithand palm pose given the closest point's surface normal. For top faces the normal is not taken from the object pointcloud but computed as the distance between bounding box center and top face center
        """
        if bounding_box_face.is_top:
            rospy.loginfo('***Top.')
            # Part I: Sample position of palm pose
            top_face_normal = bounding_box_face.center - self.bounding_box_center  # compute the normal for top faces as the vector pointing from the pointing box center to the top face center
            top_face_normal /= np.linalg.norm(top_face_normal)
            closest_point_normal = top_face_normal  # This variable is needed later, after if/else
            palm_dist_top_in_normal_direction = np.random.normal(
                self.palm_dist_to_top_face_mean,
                self.palm_dist_normal_to_obj_var
            )  # sample palm distance from normal distribution centered around mean_dist_top
            palm_position = bounding_box_face.center + palm_dist_top_in_normal_direction * top_face_normal

            # Part II: Compute vector to determine orientation of palm pose
            bounding_box_orientation_vector = bounding_box_face.orient_a[:] + bounding_box_face.orient_b[:]
            #if bounding_box_orientation_vector[1] > 0: # the y component should not be positive because the thumb will be roughly aligned with the
            #    bb_orientation_vec = -bb_orientation_vec
        else:
            # Part I: Sample the position of the palm pose
            #find the closest point in the point cloud
            m = self.segmented_object_points.shape[0]
            center_aug = np.tile(bounding_box_face.center, (m, 1))
            squared_dist = np.sum(np.square(self.segmented_object_points -
                                            center_aug),
                                  axis=1)
            min_idx = np.argmin(squared_dist)
            closest_point = self.segmented_object_points[min_idx, :]
            rospy.loginfo('Found closest point ' + str(closest_point) +
                          ' to obj_pt = ' + str(bounding_box_face.center) +
                          ' at dist = ' + str(squared_dist[min_idx]))

            # Get normal of closest point
            closest_point_normal = self.segmented_object_normals[min_idx, :]
            rospy.loginfo('Associated normal n = ' + str(closest_point_normal))
            # make sure the normal actually points outwards
            center_to_face = bounding_box_face.center - self.bounding_box_center
            if np.dot(closest_point_normal, center_to_face) < 0.:
                closest_point_normal = -closest_point_normal
            # Sample hand position
            palm_dist_side_normal_direction = np.random.normal(
                self.palm_dist_to_side_face_mean,
                self.palm_dist_normal_to_obj_var)
            palm_position = closest_point + palm_dist_side_normal_direction * closest_point_normal

            # Part II: Compute vector to determine the palm orientation
            # Pick the vertical vector with larger z value to be the palm link y axis (thumb)
            # in order to remove the orientation that runs into the table
            if abs(bounding_box_face.orient_a[2]) > abs(
                    bounding_box_face.orient_b[2]):
                bounding_box_orientation_vector = bounding_box_face.orient_a[:]
            else:
                bounding_box_orientation_vector = bounding_box_face.orient_b[:]

            if bounding_box_orientation_vector[2] < 0:
                bounding_box_orientation_vector = -bounding_box_orientation_vector

        # Add some extra 3D noise on the palm position
        palm_position = palm_position + np.random.normal(
            0, self.palm_position_3D_sample_var, 3)

        # Add some noise to the palm orientation
        bounding_box_orientation_vector = bounding_box_orientation_vector + np.random.normal(
            0., self.wrist_roll_orientation_var, 3)
        bounding_box_orientation_vector /= np.linalg.norm(
            bounding_box_orientation_vector)

        # Sample orientation
        palm_orientation_quat = self.sample_palm_orientation(
            closest_point_normal, bounding_box_orientation_vector)

        # Put this into a Stamped Pose for publishing
        palm_pose = PoseStamped()
        palm_pose.header.frame_id = 'world'
        palm_pose.header.stamp = rospy.Time.now()
        palm_pose.pose.position = palm_position
        palm_pose.pose.orientation = palm_orientation_quat
        rospy.loginfo('Chosen palm position is ' +
                      str(palm_pose.pose.position))
        return palm_pose

    def publish_bounding_box_faces_center_points(self, bounding_box_faces):
        # Create stamped poses for faces center position
        face_centers_world_frame = []
        center_stamped_world = PointStamped()
        center_stamped_world.header.frame_id = 'world'
        for _, face in enumerate(bounding_box_faces):
            center_stamped_world.point.x = face.center[0]
            center_stamped_world.point.y = face.center[1]
            center_stamped_world.point.z = face.center[2]
            face_centers_world_frame.append(
                copy.deepcopy(center_stamped_world))

        #They also publish the points, not sure why needed: publish_points(self.grasp_pose_pub, face_centers_world_frame)

    def get_axis_aligned_bounding_box_faces(self):
        """ Get the center points of 3 axis aligned bounding box faces, 1 top and 2 closest to camera

        Black, R, G, B --> Bounding Box Corner Points 1,2,3,4. Note: Assumes axis-aligned bounding box, needs to be adapted for oriented bounding boxes. 
        """
        # The 1. and 5. point of bounding_box_corner points are cross-diagonal
        # Also the bounding box axis are aligned to the world frame
        x_axis, y_axis, z_axis = np.array([1, 0, 0]), np.array(
            [0, 1, 0]), np.array([0, 0, 1])
        size_x, size_y, size_z = self.object_size[0], self.object_size[
            1], self.object_size[2]
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
        # If the object is too short, only select top grasps.
        rospy.loginfo('##########################')
        rospy.loginfo('Obj_height: %s' % size_z)
        if size_z < self.min_object_height:
            rospy.loginfo('Object is short, only use top grasps!')
            return [faces_world_frame[0]]

        return faces_world_frame

    def sample_grasp_preshape(self):
        """ Grasp preshape service callback for sampling Hithand grasp preshapes.
        """
        response = GraspPreshapeResponse()
        # Compute bounding box faces
        bounding_box_faces = self.get_axis_aligned_bounding_box_faces()
        self.palm_goal_pose_world = []
        for i in xrange(len(bounding_box_faces)):
            # Get the desired palm pose given the point from the bounding box
            palm_pose_world = self.find_palm_pose_from_bounding_box_centers(
                bounding_box_faces)
            self.palm_goal_pose_world.append(palm_pose_world)

            self.set_palm_rand_pose_limits(palm_pose_world)
            for j in xrange(self.num_samples_per_preshape):
                sampled_palm_pose = self.sample_uniformly_around_preshape_palm_pose(
                    palm_pose_world.header.frame_id)
                response.palm_goal_pose_world.append(sampled_palm_pose)
                # Sample remaining joint values
                hithant_joint_state = self.

    def generate_grasp_preshape(self):
        res = GraspPreshapeResponse()
        return res

    def handle_generate_hithand_preshape(self, req):
        """ Handler for the advertised service. Decides whether the hand preshape should be sampled or generated
        """
        self.update_object_information(
        )  # Get new information on segmented object from rostopics/disk and store in instance attributes
        if req.sample:
            return self.sample_grasp_preshape()
        else:
            return self.generate_grasp_preshape()

    def create_hithand_preshape_server(self):
        rospy.Service('generate_hithand_preshape_service', GraspPreshape,
                      self.handle_generate_hithand_preshape)
        rospy.loginfo('Service generate_hithand_preshape:')
        rospy.loginfo('Ready to generate the grasp preshape.')


if __name__ == '__main__':
    ghp = GenerateHithandPreshape()
    ghp.create_hithand_preshape_server()
    #ghp.get_axis_aligned_bounding_box_faces()

    rospy.spin()