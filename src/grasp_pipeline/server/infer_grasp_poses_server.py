#!/usr/bin/env python
# works with FFHNet tag of ICRA2022
import numpy as np
import rospy
import tf.transformations as tft
import os
import sys

use_gan = False
# TODO: gan evaluator is somewhere wrong.
if use_gan:
    sys.path.insert(0,'/home/yb/workspace/Multifinger-Net-dev')
    from FFHNet.models.ffhgan import FFHGANet
    from FFHNet.config.config import Config

use_new_config = rospy.get_param('use_new_config')
if use_new_config:
    # Add FFHNet to the path
    sys.path.append(rospy.get_param('ffhnet_path'))
    from FFHNet.models.ffhnet import FFHNet
    from FFHNet.config.config import Config
else:
    # Add FFHNet to the path
    sys.path.insert(0,rospy.get_param('ffhnet_path'))
    from FFHNet.models.ffhnet import FFHNet
    from FFHNet.config.eval_config import EvalConfig

from FFHNet.utils import visualization

from geometry_msgs.msg import PoseStamped
from grasp_pipeline.srv import *
from grasp_pipeline.utils import utils
from sensor_msgs.msg import JointState


class GraspInference():
    def __init__(self):
        rospy.init_node('grasp_inference_node')
        if use_new_config:
            config_path = '/data/hdd1/qf/ffhflow_model_history/ffhnet-prior-vae/config.yaml'
            config = Config(config_path)
            cfg = config.parse()
        elif use_gan:
            config_path = '/home/yb/workspace/Multifinger-Net-dev/models/ffhgenerator/hparams.yaml'
            config = Config(config_path)
            cfg = config.parse()

        else:
            cfg = EvalConfig().parse()
        self.load_path = rospy.get_param('ffhnet_model_path')
        if use_new_config:
            self.FFHNet.load_ffhgenerator(epoch=10,
                        load_path='/data/hdd1/qf/ffhflow_model_history/ffhnet-prior-vae')
        elif use_gan:
            self.FFHGAN = FFHGANet(cfg)
            self.FFHGAN.load_ffhgenerator(epoch=25,
                        load_path='/home/yb/workspace/Multifinger-Net-dev/models/ffhgenerator/')
        else:
            self.FFHNet = FFHNet(cfg)
            self.FFHNet.load_ffhgenerator(epoch=10,
                                        load_path=os.path.join(self.load_path,'models/ffhgenerator'))

        self.FFHNet.load_ffhevaluator(
            epoch=30,
            load_path=os.path.join(self.load_path, 'models/ffhevaluator'))
        self.VISUALIZE = rospy.get_param('visualize', False)

    def build_pose_list(self, rot_matrix, transl, frame_id='object_centroid_vae'):
        assert rot_matrix.shape[1:] == (
            3, 3), "Assumes palm rotation is 3*3 matrix."
        assert rot_matrix.shape[0] == transl.shape[
            0], "Batch dimension of rot and trans not equal."

        poses = []

        for i in range(rot_matrix.shape[0]):
            rot_hom = utils.hom_matrix_from_rot_matrix(rot_matrix[i, :, :])
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
        add_offset = False
        fix_joint = False
        joint_confs = []
        offset = 0.2
        joint_offset = [0,offset,offset,0,offset,offset,0,offset,offset,0,offset,offset,0,offset,offset]

        for i in range(joint_conf.shape[0]):
            jc = JointState()
            joint_conf_i = joint_conf[i, :]
            if add_offset:
                joint_conf_i += np.asarray(joint_offset)
            elif fix_joint:
                joint_conf_i = np.asarray(joint_offset)

            jc.position = joint_conf_i
            joint_confs.append(jc)
        return joint_confs

    def handle_infer_grasp_poses(self, req):
        # reshape
        bps_object = np.load(rospy.get_param('object_pcd_enc_path'))
        n_samples = req.n_poses
        results = self.FFHNet.generate_grasps(
            bps_object, n_samples=n_samples, return_arr=True)


        if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)

        # prepare response
        res = InferGraspPosesResponse()
        res.palm_poses = self.build_pose_list(
            results['rot_matrix'], results['transl'])
        res.joint_confs = self.build_joint_conf_list(results['joint_conf'])

        return res

    def to_grasp_dict(self, palm_poses, joint_confs, probs=False):
        """Take the palm_poses and joint_confs in ros-format and convert them to a dict with two 3D arrays in order to
        use as input for FFHEvaluator

        Args:
            palm_poses (List of PoseStamped): List of Palm poses in object frame
            joint_confs (List of JointState): List of joint states belonging to grasp

        Returns:
            grasp_dict (dict): k1 is
        """
        # prepare
        batch_n = len(palm_poses)
        joint_arr = np.zeros((batch_n, 15))
        rot_matrix_arr = np.zeros((batch_n, 3, 3))
        transl_arr = np.zeros((batch_n, 3))
        prob_arr = np.zeros((batch_n, 1))
        # convert
        for i, (palm_pose, joint_conf, prob) in enumerate(zip(palm_poses, joint_confs, probs)):
            q = palm_pose.pose.orientation
            t = palm_pose.pose.position

            rot_matrix_arr[i, :, :] = tft.quaternion_matrix(
                [q.x, q.y, q.z, q.w])[:3, :3]
            transl_arr[i, :] = [t.x, t.y, t.z]
            joint_arr[i, :] = np.array(joint_conf.position)
            prob_arr[i] = prob

        # Build grasp dict
        grasp_dict = {}
        grasp_dict['rot_matrix'] = rot_matrix_arr
        grasp_dict['transl'] = transl_arr
        grasp_dict['joint_conf'] = joint_arr
        grasp_dict['log_prob'] = prob_arr
        return grasp_dict

    def handle_evaluate_and_filter_grasp_poses(self, req):
        bps_object = np.load(rospy.get_param('object_pcd_enc_path'))
        grasp_dict = self.to_grasp_dict(req.palm_poses, req.joint_confs, req.probs)
        filter_with_prob = True
        print('evaluator: filter_with_prob:',filter_with_prob)
        try:
            results = self.FFHNet.filter_grasps(
                bps_object, grasp_dict, thresh=req.thresh, filter_with_prob=filter_with_prob)
        except Exception:
            print('here!!!!!!!!!!!!!!!!')
            print(grasp_dict)
            print(req.thresh)
            results = grasp_dict

        n_grasps_filt = results['rot_matrix'].shape[0]

        palm_poses = req.palm_poses
        n_samples = len(palm_poses)
        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" %
              (n_grasps_filt / n_samples))

        if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)

        # prepare response
        res = EvaluateAndFilterGraspPosesResponse()
        res.palm_poses = self.build_pose_list(
            results['rot_matrix'], results['transl'])
        res.joint_confs = self.build_joint_conf_list(results['joint_conf'])

        return res

    def handle_evaluate_grasp_poses(self, req):
        bps_object = np.load(rospy.get_param('object_pcd_enc_path'))
        # Build a dict with all the grasps
        p_success = self.FFHNet.evaluate_grasps(
            bps_object, grasps, return_arr=True)

    def create_infer_grasp_poses_server(self):
        rospy.Service('infer_grasp_poses', InferGraspPoses,
                      self.handle_infer_grasp_poses)
        rospy.loginfo('Service infer_grasp_poses')
        rospy.loginfo('Ready to sample grasps from FFHGenerator.')

    def create_evaluate_grasp_poses_server(self):
        rospy.Service('evaluate_grasp_poses', EvaluateGraspPoses,
                      self.handle_evaluate_grasp_poses)
        rospy.loginfo('Service evaluate_grasp_poses')
        rospy.loginfo('Ready to evaluate grasps with the FFHEvaluator.')

    def create_evaluate_and_filter_grasp_poses_server(self):
        rospy.Service('evaluate_and_filter_grasp_poses', EvaluateAndFilterGraspPoses,
                      self.handle_evaluate_and_filter_grasp_poses)
        rospy.loginfo('Service evaluate_and_filter_grasp_poses')
        rospy.loginfo(
            'Ready to evaluate and filter grasps with the FFHEvaluator.')


if __name__ == '__main__':
    gi = GraspInference()
    gi.create_infer_grasp_poses_server()
    gi.create_evaluate_and_filter_grasp_poses_server()
    rospy.spin()