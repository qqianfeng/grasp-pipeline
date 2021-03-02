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
        self.grasp_file_name = self.data_recording_path + 'grasp_data_utah.h5'

    def handle_record_grasp_data_utah(self, req):
        with h5py.File(self.grasp_file_name, 'a') as hdf:
            grasp_name = 'grasp_' + str(req.grasp_id).zfill(6)

            hdf.create_dataset(grasp_name + '_config_obj', data=req.grasp_config_obj)
            hdf.create_dataset(grasp_name + '_sparse_voxel',
                               data=np.reshape(req.sparse_voxel, [len(req.sparse_voxel) / 3, 3]))
            hdf.create_dataset(grasp_name + '_dim_w_h_d', data=req.dim_w_h_d)
            hdf.create_dataset(grasp_name + '_label', data=req.label)

        res = SimGraspDataResponse(success=True)
        return res

    def create_record_data_server(self):
        rospy.Service('record_grasp_data_utah', SimGraspData, self.handle_record_grasp_data_utah)
        rospy.loginfo('Service record_grasp_data_utah:')
        rospy.loginfo('Ready to record grasp data.')


if __name__ == '__main__':
    record_grasp_data = RecordGraspData()
    record_grasp_data.create_record_data_server()
    rospy.spin()
