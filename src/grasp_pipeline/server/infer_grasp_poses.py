import numpy as np
import rospy
import tf.transformations as tft
import torch

from hithand_grabnet.models.grabnet import GrabNet
from hithand_grabnet.config.eval_config import EvalConfig
from geometry_msgs.msg import PoseStamped
from grasp_pipeline.srv import *
from grasp_pipeline.utils import utils
from sensor_msgs.msg import JointState


class GraspInference():
    def __init__(self):
        rospy.init_node('grasp_inference_node')
        cfg = EvalConfig().parse()
        self.grabnet = GrabNet(cfg)
        self.grabnet.load_vae(epoch=10, is_train=False)

    def build_pose_list(self, rot_mat, transl, frame_id='centroid_vae'):
        assert rot_mat.shape[1:] == (3, 3), "Assumes palm rotation is 3*3 matrix."
        assert rot_mat.shape[0] == transl.shape[0], "Batch dimension of rot and trans not equal."

        poses = []

        for i in range(rot_mat.shape[0]):
            rot_hom = utils.hom_matrix_from_rot_matrix(rot_mat[i, :, :])
            quat = tft.quaternion_from_matrix(rot_hom)
            t = transl[i, :]

            pose_st = PoseStamped()
            pose_st.header.frame_id = frame_id
            pose_st.pose.position.x = t[0]
            pose_st.pose.position.y = t[1]
            pose_st.pose.position.z = t[2]
            pose_st.pose.orientation.x = quat[0]
            pose_st.pose.orientation.y = quat[1]
            pose_st.pose.orientation.z = quat[2]
            pose_st.pose.orientation.w = quat[3]

            poses.append(pose_st)

        return poses

    def build_joint_conf_list(self, joint_conf):
        joint_confs = []
        for i in range(joint_conf.shape[0]):
            jc = JointState()
            jc.position = joint_conf[i, :]
            joint_confs.append(jc)
        return joint_confs

    def handle_infer_grasp_poses(self, req):
        # reshape
        bps_object = np.load('/home/vm/pcd_enc.npy')
        n_samples = 5
        results = self.grabnet.sample_grasps(bps_object, n_samples=n_samples, return_arr=True)

        # prepare response
        res = InferGraspsResponse()
        res.palm_poses = self.build_pose_list(results['rot_mat'], results['transl'])
        res.joint_confs = self.build_joint_conf_list(results['joint_conf'])

        return res

    def create_infer_grasp_poses_server(self):
        rospy.Service('infer_grasp_poses', InferGraspPoses, self.handle_infer_grasp_poses)
        rospy.loginfo('Service infer_grasp_poses')
        rospy.loginfo('Ready to sample grasps from VAE model.')


if __name__ == '__main__':
    gi = GraspInference()
    gi.handle_infer_grasp_poses(True)
    gi.create_infer_grasp_poses_server()
    rospy.spin()
