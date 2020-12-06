#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *


class GenerateHithandPreshape():
    def __init__(self):
        rospy.init_node("generate_hithand_preshape_node")

    def sample_grasp_preshape(self, req):
        """ Grasp preshape service callback for sampling grasp preshapes.
        """
        res = GraspPreshapeResponse()
        # Compute bounding box faces

    def generate_grasp_preshape(self, req):
        res = GraspPreshapeResponse()
        return res

    def handle_generate_hithand_preshape(self, req):
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