import rospy
from grasp_pipeline.srv import *
from geometry_msgs.msg import Pose, Quaternion
import h5py
import numpy as np


class RecordGraspData():
    def __init__(self):
        rospy.init_node('record_grasp_data_node')
        self.num_grasps_per_object = rospy.get_param('num_grasps_per_object')
        self.data_recording_path = rospy.get_param('data_recording_path')
        self.grasp_data_file_name = self.data_recording_path + 'grasp_data.h5'
        self.hand_joint_state_name = [
            'Right_Index_0', 'Right_Index_1', 'Right_Index_2', 'Right_Index_3', 'Right_Little_0',
            'Right_Little_1', 'Right_Little_2', 'Right_Little_3', 'Right_Middle_0',
            'Right_Middle_1', 'Right_Middle_2', 'Right_Middle_3', 'Right_Ring_0', 'Right_Ring_1',
            'Right_Ring_2', 'Right_Ring_3', 'Right_Thumb_0', 'Right_Thumb_1', 'Right_Thumb_2',
            'Right_Thumb_3'
        ]

        self.initialize_data_file()

    def initialize_data_file(self):
        # a: Read/write if exists, create otherwise
        grasp_file = h5py.File(self.grasp_data_file_name, 'a')
        hand_js_name_key = 'hand_joint_state_name'
        if hand_js_name_key not in grasp_file:
            grasp_file.create_dataset(hand_js_name_key, data=self.hand_joint_state_name)

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
        # r+ : Read/write, file must exist
        grasp_file = h5py.File(self.grasp_data_file_name, 'r+')

    def create_record_grasp_data_server(self):
        rospy.Service('record_gras_data', RecordGraspDataSim, self.handle_record_grasp_data)
        rospy.loginfo('Service record_grasp_data:')
        rospy.loginfo('Ready to record your awesome grasp data.')


if __name__ == '__main__':
    rgd = RecordGraspData()
    rgd.create_record_grasp_data_server()
    rospy.spin()
