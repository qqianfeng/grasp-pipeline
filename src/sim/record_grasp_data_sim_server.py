import rospy
from grasp_pipeline.srv import *
from geometry_msgs.msg import Pose, Quaternion
import h5py
import numpy as np
from datetime import datetime


class RecordGraspData():
    def __init__(self):
        rospy.init_node('record_grasp_data_node')
        self.num_grasps_per_object = rospy.get_param('num_grasps_per_object', 5)
        self.data_recording_path = rospy.get_param('data_recording_path', '/home/vm/')
        self.grasp_data_file_name = self.data_recording_path + 'grasp_data.h5'
        self.hand_joint_state_name = [
            'Right_Index_0', 'Right_Index_1', 'Right_Index_2', 'Right_Index_3',     \
            'Right_Little_0', 'Right_Little_1', 'Right_Little_2', 'Right_Little_3', \
            'Right_Middle_0', 'Right_Middle_1', 'Right_Middle_2', 'Right_Middle_3', \
            'Right_Ring_0', 'Right_Ring_1', 'Right_Ring_2', 'Right_Ring_3',         \
            'Right_Thumb_0', 'Right_Thumb_1', 'Right_Thumb_2', 'Right_Thumb_3'
        ]
        self.recording_session_id = None
        self.initialize_data_file()

    def convert_pose_to_list(self, pose):
        return [
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
            pose.pose.orientation.w
        ]

    def initialize_sess_metadata(self, gp_curr_sess_metadata):
        gp_curr_sess_metadata.create_dataset("sess_start", data=datetime.now().isoformat())
        gp_curr_sess_metadata.create_dataset("sess_num_grasps", data=0, dtype='u4')
        gp_curr_sess_metadata.create_dataset("sess_num_top_grasps", data=0, dtype='u4')
        gp_curr_sess_metadata.create_dataset("sess_num_success_grasps", data=0, dtype='u4')
        gp_curr_sess_metadata.create_dataset("sess_num_failure_grasps", data=0, dtype='u4')

    def initialize_data_file(self):
        """ Creates a new grasp file under grasp_data_file_name or opens existing one and then creates "grasp_trials" and "grasp_metadata" groups if they do not yet exist.
        """
        # a: Read/write if exists, create otherwise
        with h5py.File(self.grasp_data_file_name, 'a') as file:
            # Either create the whole file structure if file did not exist before:
            if 'recording_sessions' not in file.keys():
                # grasp_data.h5 --> recording_sessions Create recording_sessions group
                gp_recording_sessions = file.create_group('recording_sessions')
                # recording_sessions --> recording_session1
                self.curr_sess_name = 'recording_session_0001'
                gp_recording_session_1 = gp_recording_sessions.create_group(self.curr_sess_name)
                # recording_sessions1 --> (metadata and grasp_trials)
                sess_metadata = gp_recording_session_1.create_group('metadata')
                gp_recording_session_1.create_group('grasp_trials')
                # Initialize metadata for current session
                self.initialize_sess_metadata(sess_metadata)

                # grasp_data.h5 --> metadata Create overall metadata
                gp_metadata = file.create_group('metadata')
                # Initalize all the variables to be stored in metadata
                gp_metadata.create_dataset("datetime_recording_start",
                                           data=[datetime.now().isoformat()],
                                           maxshape=(None, ))  # print(grasps.dtype)
                gp_metadata.create_dataset("total_num_grasps", data=0, dtype='u4')
                gp_metadata.create_dataset("total_num_top_grasps", data=0, dtype='u4')
                gp_metadata.create_dataset("total_num_success_grasps", data=0, dtype='u4')
                gp_metadata.create_dataset("total_num_failure_grasps", data=0, dtype='u4')
                gp_metadata.create_dataset("total_num_recordings", data=1, dtype='u4')

                # set current sess name
                self.curr_sess_name = 'recording_session_0001'

            # If recording_sessions group exists, just create a new subgroup
            elif 'recording_sessions' in file.keys():
                file['metadata']['total_num_recordings'][()] += 1
                self.curr_sess_name = 'recording_session_' + str(
                    file['metadata']['total_num_recordings'][()]).zfill(4)
                dset_recording_start = file["metadata"]["datetime_recording_start"]
                dset_recording_start.resize((dset_recording_start.shape[0] + 1, ))
                dset_recording_start[-1] = datetime.now().isoformat()

                # recording_sessions --> recordings_session_name
                gp_recording_session_n = file['recording_sessions'].create_group(
                    self.curr_sess_name)
                # recording_sessions_name --> (metadata and grasp_trials)
                sess_metadata = gp_recording_session_n.create_group('metadata')
                gp_recording_session_n.create_group('grasp_trials')
                # Initialize metadata for current session
                self.initialize_sess_metadata(sess_metadata)

    def update_grasp_metadata(self, metadata_group, curr_sess_metadata_group, is_top_grasp,
                              grasp_success_label):
        # Overall meta data and grasp trial metadata
        metadata_group['total_num_grasps'][()] += 1
        curr_sess_metadata_group['sess_num_grasps'][()] += 1
        if is_top_grasp:
            metadata_group['total_num_top_grasps'][()] += 1
            curr_sess_metadata_group['sess_num_top_grasps'][()] += 1
        if grasp_success_label:
            metadata_group['total_num_success_grasps'][()] += 1
            curr_sess_metadata_group['sess_num_success_grasps'][()] += 1
        else:
            metadata_group['total_num_failure_grasps'][()] += 1
            curr_sess_metadata_group['sess_num_failure_grasps'][()] += 1

    def handle_record_grasp_data(self, req):
        # r+ : Read/write, file must exist
        with h5py.File(self.grasp_data_file_name, 'r+') as grasp_file:
            grasp_file = h5py.File(self.grasp_data_file_name, 'r+')
            # workflow: Create new group under grasp trials. Create dataset for each piece of data stored for this trial
            # 1. Update grasp meta information and update sess metadata
            self.update_grasp_metadata(
                grasp_file['metadata'],
                grasp_file['recording_sessions'][self.curr_sess_name]['metadata'],
                req.is_top_grasp, req.grasp_success_label)
            grasp_trial_id_str = str(grasp_file['metadata']['total_num_grasps'][()]).zfill(6)

            # 2. Create new group under curr sess grasp_trials with the name e.g. grasp_000005 if 5th grasp
            grasp_group = grasp_file['recording_sessions'][
                self.curr_sess_name]['grasp_trials'].create_group('grasp_' + grasp_trial_id_str)

            # 3. Add all the grasp-specific data and store in datasets under new group
            # object_name
            grasp_group.create_dataset('object_name', data=req.object_name)
            # time stamp
            grasp_group.create_dataset('time_stamp', data=req.time_stamp)
            # is_top_grasp
            grasp_group.create_dataset('is_top_grasp', data=req.is_top_grasp)
            # grasp_success_label
            grasp_group.create_dataset('grasp_success_label', data=req.grasp_success_label)
            # object_size
            grasp_group.create_dataset('object_size', data=req.object_size)
            # sparse_voxel_grid
            voxel_grid = np.reshape(req.sparse_voxel_grid, [len(req.sparse_voxel_grid) / 3, 3])
            grasp_group.create_dataset('sparse_voxel_grid', data=voxel_grid)

            # 4. Record...

            # 5. Record all poses
            # object_world_sim_pose
            grasp_group.create_dataset('object_world_sim_pose',
                                       data=self.convert_pose_to_list(req.object_world_sim_pose))
            # object_world_seg_pose
            grasp_group.create_dataset('object_world_seg_pose',
                                       data=self.convert_pose_to_list(req.object_world_seg_pose))
            # desired_preshape_palm_world_pose
            grasp_group.create_dataset('desired_preshape_palm_world_pose',
                                       data=self.convert_pose_to_list(
                                           req.desired_preshape_palm_world_pose))
            # true_preshape_palm_world_pose
            grasp_group.create_dataset('true_preshape_palm_world_pose',
                                       data=self.convert_pose_to_list(
                                           req.true_preshape_palm_world_pose))
            # closed_palm_world_pose
            grasp_group.create_dataset('closed_palm_world_pose',
                                       data=self.convert_pose_to_list(req.closed_palm_world_pose))
            # lifted_palm_world_pose
            grasp_group.create_dataset('lifted_palm_world_pose',
                                       data=self.convert_pose_to_list(req.lifted_palm_world_pose))

            #6. Record all joint states
            #desired_preshape_hithand_joint_state
            grasp_group.create_dataset('desired_preshape_hithand_joint_state',
                                       data=req.desired_preshape_hithand_joint_state.position)
            #true_preshape_hithand_joint_state
            grasp_group.create_dataset('true_preshape_hithand_joint_state',
                                       data=req.true_preshape_hithand_joint_state.position)
            #closed_hithand_joint_state
            grasp_group.create_dataset('closed_hithand_joint_state',
                                       data=req.closed_hithand_joint_state.position)
            #lifted_hithand_joint_state
            grasp_group.create_dataset('lifted_hithand_joint_state',
                                       data=req.lifted_hithand_joint_state.position)

            #8. Return response, context manager closes file
            res = RecordGraspDataSimResponse()
            res.success = True
            return res

    def create_record_grasp_data_server(self):
        rospy.Service('record_grasp_data', RecordGraspDataSim, self.handle_record_grasp_data)
        rospy.loginfo('Service record_grasp_data:')
        rospy.loginfo('Ready to record your awesome grasp data.')


if __name__ == '__main__':
    rgd = RecordGraspData()
    rgd.create_record_grasp_data_server()
    rospy.spin()
