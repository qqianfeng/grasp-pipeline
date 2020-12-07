#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from std_msgs.msg import Float32MultiArray
import open3d as o3d
from tf import TransformListener


class GenerateHithandPreshape():
    def __init__(self):
        rospy.init_node("generate_hithand_preshape_node")
        object_segmented_size_subscriber = rospy.Subscriber(
            "/object_segmented_size")
        self.current_object_size = None
        self.current_object_bounding_box_corner_points = None
        self.segmented_object_pcd = None

    def update_object_information(self):
        """ Update instance variables related to the object of interest

        This is intended to 1.) receive a single message from the segmented_object topics and store them in instance attributes and 
        2.) read the segmented object point cloud from disk
        """
        self.current_object_size = rospy.wait_for_message(
            '/segmented_object_size', Float32MultiArray).data
        self.current_object_bounding_box_corner_points = rospy.wait_for_message(
            '/segmented_object_bounding_box_corner_points',
            Float32MultiArray).data
        self.segmented_object_pcd = o3d.io.read_point_cloud(
            '/home/vm/object.pcd')

    def get_bounding_box_faces_and_center(self):
        pass

    def sample_grasp_preshape(self, req):
        """ Grasp preshape service callback for sampling grasp preshapes.
        """
        res = GraspPreshapeResponse()
        # Compute bounding box faces

    def generate_grasp_preshape(self, req):
        res = GraspPreshapeResponse()
        return res

    def handle_generate_hithand_preshape(self, req):
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
    ghp.create_hithand_preshape_server()

    rospy.spin()