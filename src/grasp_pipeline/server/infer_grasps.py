import numpy as np
import rospy
import torch

from hithand_grabnet.models.grabnet import GrabNet
from hithand_grabnet.config.eval_config import EvalConfig
from grasp_pipeline.srv import *


class GraspInference():
    def __init__(self):
        rospy.init_node('grasp_inference_node')
        cfg = EvalConfig().parse()
        self.grabnet = GrabNet(cfg)
        self.grabnet.load_vae(epoch=-1, is_train=False)

    def build_pose_list(self, palm_rot, palm_trans):
        raise NotImplementedError

    def build_joint_conf_list(self, joint_conf):
        raise NotImplementedError

    def handle_infer_grasps(self, req):
        # reshape
        results = self.grabnet.sample_grasps(req.bps_object, n_samples=req.n_samples)

        # prepare response
        res = InferGraspsResponse()
        res.palm_poses = self.build_pose_list(results['rot_mat'], results['transl'])
        res.joint_confs = self.build_joint_conf_list(results['joint_confs'])

    def create_infer_grasps_server(self):
        rospy.Service('infer_grasps', InferGrasps, self.handle_infer_grasps)
        rospy.loginfo('Service infer_graps')
        rospy.loginfo('Ready to sample grasps from VAE model.')


if __name__ == '__main__':
    gi = GraspInference()
    gi.create_infer_grasps_server()
    rospy.spin()
