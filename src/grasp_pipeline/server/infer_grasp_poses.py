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
        self.grabnet.load_vae(epoch=10, is_train=False)

    def build_pose_list(self, palm_rot, palm_trans):
        raise NotImplementedError

    def build_joint_conf_list(self, joint_conf):
        raise NotImplementedError

    def handle_infer_grasp_poses(self, req):
        # reshape
        bps_object = np.load('/home/vm/pcd_enc.npy')
        n_samples = 5
        results = self.grabnet.sample_grasps(bps_object, n_samples=n_samples)

        # CONTINUE HERE, TURN RESPONSE OF MODEL INTO ROS STUFF
        CONTINUE HERE
        # prepare response
        res = InferGraspsResponse()
        res.palm_poses = self.build_pose_list(results['rot_6D'], results['transl'])
        res.joint_confs = self.build_joint_conf_list(results['joint_confs'])

    def create_infer_grasp_poses_server(self):
        rospy.Service('infer_grasp_poses', InferGraspPoses, self.handle_infer_grasp_poses)
        rospy.loginfo('Service infer_grasp_poses')
        rospy.loginfo('Ready to sample grasps from VAE model.')


if __name__ == '__main__':
    gi = GraspInference()
    gi.handle_infer_grasp_poses(True)
    gi.create_infer_grasp_poses_server()
    rospy.spin()
