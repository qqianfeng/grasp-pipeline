#!/usr/bin/env python

import roslib
roslib.load_manifest('grasp_pipeline')
import rospy
from grasp_pipeline.srv import *
from geometry_msgs.msg import Pose, Quaternion
from sensor_msgs.msg import JointState, CameraInfo
import tf
import numpy as np
import h5py


class RecordGraspData():
    def __init__(self):
        rospy.init_node('record_grasp_data_server')
        self.num_grasps_per_object = rospy.get_param('~num_grasps_per_object', 10)
        self.data_recording_path = rospy.get_param('~data_recording_path', '/home/vm/utah/')
        self.grasp_file_name = self.data_recording_path + 'grasp_data.h5'
        self.initialize_data_file()

    def initialize_data_file(self):
        #a: Read/write if exists, create otherwise (default)
        grasp_file = h5py.File(self.grasp_file_name, 'a')
        hand_joint_state_name = [
            'index_joint_0', 'index_joint_1', 'index_joint_2', 'index_joint_3', 'middle_joint_0',
            'middle_joint_1', 'middle_joint_2', 'middle_joint_3', 'ring_joint_0', 'ring_joint_1',
            'ring_joint_2', 'ring_joint_3', 'thumb_joint_0', 'thumb_joint_1', 'thumb_joint_2',
            'thumb_joint_3'
        ]
        hand_js_name_key = 'hand_joint_state_name'
        if hand_js_name_key not in grasp_file:
            grasp_file.create_dataset(hand_js_name_key, data=hand_joint_state_name)

        max_object_id_key = 'max_object_id'
        if max_object_id_key not in grasp_file:
            grasp_file.create_dataset(max_object_id_key, data=-1)
        cur_object_name_key = 'cur_object_name'
        if cur_object_name_key not in grasp_file:
            grasp_file.create_dataset(cur_object_name_key, data='empty')

        total_grasps_num_key = 'total_grasps_num'
        if total_grasps_num_key not in grasp_file:
            grasp_file.create_dataset(total_grasps_num_key, data=0)
        suc_grasps_num_key = 'suc_grasps_num'
        if suc_grasps_num_key not in grasp_file:
            grasp_file.create_dataset(suc_grasps_num_key, data=0)

        grasp_file.close()

    def handle_record_grasp_data(self, req):
        #'r+': Read/write, file must exist
        grasp_file = h5py.File(self.grasp_file_name, 'r+')
        if req.grasp_id == 0:
            grasp_file['max_object_id'][()] += 1
            grasp_file['cur_object_name'][()] = req.object_name

        #object_id = grasp_file['max_object_id'][()] + 1
        object_id = grasp_file['max_object_id'][()]
        object_grasp_id = 'object_' + str(object_id) + '_grasp_' + str(req.grasp_id)

        global_grasp_id = grasp_file['total_grasps_num'][()]
        obj_grasp_id_key = 'grasp_' + str(global_grasp_id) + '_obj_grasp_id'
        if obj_grasp_id_key not in grasp_file:
            grasp_file.create_dataset(obj_grasp_id_key, data=object_grasp_id)

        grasp_object_name_key = 'object_' + str(object_id) + '_name'
        if grasp_object_name_key not in grasp_file:
            grasp_file.create_dataset(grasp_object_name_key, data=req.object_name)

        voxel_grid_key = object_grasp_id + '_sparse_voxel_grid'
        if voxel_grid_key not in grasp_file:
            voxel_grid = np.reshape(req.sparse_voxel_grid, [len(req.sparse_voxel_grid) / 3, 3])
            grasp_file.create_dataset(voxel_grid_key, data=voxel_grid)

        object_size_key = object_grasp_id + '_object_size'
        if object_size_key not in grasp_file:
            grasp_file.create_dataset(object_size_key, data=req.object_size)

        preshape_palm_world_pose_list = [
            req.preshape_palm_world_pose.pose.position.x,
            req.preshape_palm_world_pose.pose.position.y,
            req.preshape_palm_world_pose.pose.position.z,
            req.preshape_palm_world_pose.pose.orientation.x,
            req.preshape_palm_world_pose.pose.orientation.y,
            req.preshape_palm_world_pose.pose.orientation.z,
            req.preshape_palm_world_pose.pose.orientation.w
        ]
        palm_world_pose_key = object_grasp_id + '_preshape_palm_world_pose'
        if palm_world_pose_key not in grasp_file:
            grasp_file.create_dataset(palm_world_pose_key, data=preshape_palm_world_pose_list)

        true_preshape_palm_world_pose_list = [
            req.true_preshape_palm_world_pose.pose.position.x,
            req.true_preshape_palm_world_pose.pose.position.y,
            req.true_preshape_palm_world_pose.pose.position.z,
            req.true_preshape_palm_world_pose.pose.orientation.x,
            req.true_preshape_palm_world_pose.pose.orientation.y,
            req.true_preshape_palm_world_pose.pose.orientation.z,
            req.true_preshape_palm_world_pose.pose.orientation.w
        ]
        palm_world_pose_key = object_grasp_id + '_true_preshape_palm_world_pose'
        if palm_world_pose_key not in grasp_file:
            grasp_file.create_dataset(palm_world_pose_key, data=true_preshape_palm_world_pose_list)

        preshape_js_position_key = object_grasp_id + '_preshape_joint_state_position'
        if preshape_js_position_key not in grasp_file:
            grasp_file.create_dataset(preshape_js_position_key,
                                      data=req.preshape_allegro_joint_state.position)

        true_preshape_js_position_key = object_grasp_id + '_true_preshape_js_position'
        if true_preshape_js_position_key not in grasp_file:
            grasp_file.create_dataset(true_preshape_js_position_key,
                                      data=req.true_preshape_joint_state.position)

        grasp_label_key = object_grasp_id + '_grasp_label'
        if (grasp_label_key not in grasp_file) and \
                (len(req.inf_config_array) != 0):
            grasp_file.create_dataset(grasp_label_key, data=req.grasp_success_label)
            if req.grasp_success_label == 1:
                grasp_file['suc_grasps_num'][()] += 1
            grasp_file['total_grasps_num'][()] += 1

        top_grasp_key = object_grasp_id + '_top_grasp'
        if top_grasp_key not in grasp_file:
            grasp_file.create_dataset(top_grasp_key, data=req.top_grasp)

        response = SimGraspDataResponse()
        response.object_id = object_id
        response.success = True
        grasp_file.close()
        return response

    def create_record_data_server(self):
        rospy.Service('record_grasp_data', SimGraspData, self.handle_record_grasp_data)
        rospy.loginfo('Service record_grasp_data:')
        rospy.loginfo('Ready to record grasp data.')


if __name__ == '__main__':
    record_grasp_data = RecordGraspData()
    record_grasp_data.create_record_data_server()
    rospy.spin()
