#!/usr/bin/env python
import rospy
import os
import h5py
import numpy as np
from datetime import datetime

from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from sensor_msgs.msg import JointState

from grasp_pipeline.srv import *


class RecordGraspData():
    def __init__(self):
        if not DEBUG:
            rospy.init_node('record_grasp_and_collision_data_node')
        self.data_recording_path = rospy.get_param('data_recording_path')
        self.grasp_data_file_name = os.path.join(self.data_recording_path, 'grasp_data.h5')
        self.recording_session_id = None
        self.initialize_data_file()
        self.collision_id = 0
        self.no_collision_id = 0

    ###################################
    ######  PART I : Helper  ##########
    ###################################
    def convert_pose_to_list(self, pose):
        return [
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
            pose.pose.orientation.w
        ]

    ###################################
    ###  PART II : Initializations  ###
    ###################################
    def initialize_object_name_metadata_group(self, metadata_gp):
        metadata_gp.create_dataset("object_num_grasps", data=0, dtype='u4')
        metadata_gp.create_dataset("object_num_tops", data=0, dtype='u4')
        metadata_gp.create_dataset("object_num_successes", data=0, dtype='u4')
        metadata_gp.create_dataset("object_num_failures", data=0, dtype='u4')
        metadata_gp.create_dataset("object_num_collisions", data=0, dtype='u4')
        self.collision_id = {'collision':0, 'no_ik':0}
        self.no_collision_id = 0

    def initialize_sess_metadata(self, sess_metadata_gp):
        sess_metadata_gp.create_dataset("sess_start", data=datetime.now().isoformat())
        sess_metadata_gp.create_dataset("sess_num_grasps", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_tops", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_successes", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_failures", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_collisions", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_collision_to_approach_pose", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_collision_to_grasp_pose", data=0, dtype='u4')

        sess_metadata_gp.create_dataset("sess_num_grasp_pose_collide_target_object", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_grasp_pose_collide_obstacle_objects", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_close_finger_collide_obstacle_objects", data=0, dtype='u4')
        sess_metadata_gp.create_dataset("sess_num_lift_motion_moved_obstacle_objects", data=0, dtype='u4')
        
    def initialize_file_metadata(self, metadata_gp):
        metadata_gp.create_dataset("datetime_recording_start",
                                   data=[datetime.now().isoformat()],
                                   maxshape=(None, ))
        metadata_gp.create_dataset("total_num_grasps", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_tops", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_successes", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_failures", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_collisions", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_collision_to_approach_pose", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_collision_to_grasp_pose", data=0, dtype='u4')
        
        metadata_gp.create_dataset("total_num_grasp_pose_collide_target_object", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_grasp_pose_collide_obstacle_objects", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_close_finger_collide_obstacle_objects", data=0, dtype='u4')
        metadata_gp.create_dataset("total_num_lift_motion_moved_obstacle_objects", data=0, dtype='u4')
        
        metadata_gp.create_dataset("total_num_recordings", data=1, dtype='u4')

    def initialize_data_file(self):
        """ Creates a new grasp file under grasp_data_file_name or opens existing one and then creates "grasp_trials" and "grasp_metadata" groups if they do not yet exist.
        """
        # a: Read/write if exists, create otherwise
        with h5py.File(self.grasp_data_file_name, 'a') as hdf:
            # Either create the whole hdf structure if hdf did not exist before:
            if 'recording_sessions' not in hdf.keys():
                # grasp_data.h5 --> recording_sessions Create recording_sessions group
                recording_sessions_gp = hdf.create_group('recording_sessions')
                # recording_sessions --> recording_session1
                self.curr_sess_name = 'recording_session_0001'
                recording_session_1_gp = recording_sessions_gp.create_group(self.curr_sess_name)
                # recording_sessions1 --> (metadata and grasp_trials)
                sess_metadata = recording_session_1_gp.create_group('metadata')
                recording_session_1_gp.create_group('grasp_trials')
                # Initialize metadata for current session
                self.initialize_sess_metadata(sess_metadata)

                # grasp_data.h5 --> metadata Create overall metadata
                metadata_gp = hdf.create_group('metadata')

                # Initalize all the variables to be stored in metadata
                self.initialize_file_metadata(metadata_gp)

                # set current sess name
                self.curr_sess_name = 'recording_session_0001'

            # If recording_sessions group exists, just create a new subgroup
            elif 'recording_sessions' in hdf.keys():
                hdf['metadata']['total_num_recordings'][()] += 1
                self.curr_sess_name = 'recording_session_' + str(
                    hdf['metadata']['total_num_recordings'][()]).zfill(4)
                recording_start_ds = hdf["metadata"]["datetime_recording_start"]
                recording_start_ds.resize((recording_start_ds.shape[0] + 1, ))
                recording_start_ds[-1] = datetime.now().isoformat()

                # recording_sessions --> recordings_session_name
                recording_session_n_gp = hdf['recording_sessions'].create_group(
                    self.curr_sess_name)
                # recording_sessions_name --> (metadata and grasp_trials)
                sess_metadata = recording_session_n_gp.create_group('metadata')
                recording_session_n_gp.create_group('grasp_trials')
                # Initialize metadata for current session
                self.initialize_sess_metadata(sess_metadata)

    ###################################
    ######  PART III : Update  ########
    ###################################
    def update_all_metadata(self,
                            grasp_file,
                            is_top_grasp,
                            grasp_success_label,
                            collision_to_approach_pose=False,
                            collision_to_grasp_pose=False,
                            grasp_pose_collide_target_object=False,
                            grasp_pose_collide_obstacle_objects=False,
                            close_finger_collide_obstacle_objects=False,
                            lift_motion_moved_obstacle_objects=False
                            ):
        # Overall meta data and grasp trial metadata
        metadata_group = grasp_file['metadata']
        curr_sess_metadata_group = grasp_file['recording_sessions'][
            self.curr_sess_name]['metadata']

        metadata_group['total_num_grasps'][()] += 1
        curr_sess_metadata_group['sess_num_grasps'][()] += 1
        if is_top_grasp:
            metadata_group['total_num_tops'][()] += 1
            curr_sess_metadata_group['sess_num_tops'][()] += 1

        if collision_to_approach_pose:
            metadata_group['total_num_collision_to_approach_pose'][()] += 1
            curr_sess_metadata_group['sess_num_collision_to_approach_pose'][()] += 1
        if collision_to_grasp_pose:
            metadata_group['total_num_collision_to_grasp_pose'][()] += 1
            curr_sess_metadata_group['sess_num_collision_to_grasp_pose'][()] += 1
            
        # labels added for multi obj data generation
        if grasp_pose_collide_target_object:
            metadata_group['total_num_grasp_pose_collide_target_object'][()] += 1
            curr_sess_metadata_group['sess_num_grasp_pose_collide_target_object'][()] += 1
        if grasp_pose_collide_obstacle_objects:
            metadata_group['total_num_grasp_pose_collide_obstacle_objects'][()] += 1
            curr_sess_metadata_group['sess_num_grasp_pose_collide_obstacle_objects'][()] += 1
        if close_finger_collide_obstacle_objects:
            metadata_group['total_num_close_finger_collide_obstacle_objects'][()] += 1
            curr_sess_metadata_group['sess_num_close_finger_collide_obstacle_objects'][()] += 1
        if lift_motion_moved_obstacle_objects:
            metadata_group['total_num_lift_motion_moved_obstacle_objects'][()] += 1
            curr_sess_metadata_group['sess_num_lift_motion_moved_obstacle_objects'][()] += 1
            
        if grasp_success_label:
            metadata_group['total_num_successes'][()] += 1
            curr_sess_metadata_group['sess_num_successes'][()] += 1
        else:
            metadata_group['total_num_failures'][()] += 1
            curr_sess_metadata_group['sess_num_failures'][()] += 1

    def update_grasp_metadata(self,
                              metadata_group,
                              curr_sess_metadata_group,
                              is_top_grasp,
                              grasp_success_label,
                              collision_to_approach_pose=False,
                              collision_to_grasp_pose=False,
                              grasp_pose_collide_target_object=False,
                              grasp_pose_collide_obstacle_objects=False,
                              close_finger_collide_obstacle_objects=False,
                              lift_motion_moved_obstacle_objects=False
                              ):

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

        if collision_to_approach_pose:
            metadata_group['total_num_collision_to_approach_pose'][()] += 1
        if collision_to_grasp_pose:
            metadata_group['total_num_collision_to_grasp_pose'][()] += 1
            
        if grasp_pose_collide_target_object:
            metadata_group['total_num_grasp_pose_collide_target_object'][()] += 1
        if grasp_pose_collide_obstacle_objects:
            metadata_group['total_num_grasp_pose_collide_obstacle_objects'][()] += 1
        if close_finger_collide_obstacle_objects:
            metadata_group['total_num_close_finger_collide_obstacle_objects'][()] += 1
        if lift_motion_moved_obstacle_objects:
            metadata_group['total_num_lift_motion_moved_obstacle_objects'][()] += 1

    def update_all_metadata_collision(self, grasp_file, num_grasps):
        # Update overall metadata
        grasp_file['metadata']['total_num_grasps'][()] += num_grasps
        grasp_file['metadata']['total_num_collisions'][()] += num_grasps

        # Update session metadata
        rec_sess_metadata_gp = grasp_file['recording_sessions'][self.curr_sess_name]['metadata']
        rec_sess_metadata_gp['sess_num_grasps'][()] += num_grasps
        rec_sess_metadata_gp['sess_num_collisions'][()] += num_grasps  # why???

        # Update object
        object_metadata_gp = grasp_file['recording_sessions'][self.curr_sess_name]['grasp_trials'][
            self.curr_object_name]['metadata']
        object_metadata_gp['object_num_grasps'][()] += num_grasps
        object_metadata_gp['object_num_collisions'][()] += num_grasps  # why???

    def get_grasp_group(self, grasp_trials, grasp_class):
        """ Creates the entire folder structure under grasp_trials if it does not exist.
        Returns the 
        """
        object_name = self.curr_object_name

        # If this object name is not under grasp_trials already, create folder structure
        if object_name not in grasp_trials.keys():
            # grasp_trials --> object_name
            object_gp = grasp_trials.create_group(object_name)
            # object_name --> metadata
            metadata_gp = object_gp.create_group('metadata')
            self.initialize_object_name_metadata_group(metadata_gp)
            # object_name --> grasps
            grasps_gp = object_gp.create_group('grasps')
            # grasps --> collision/ no-collision
            grasps_gp.create_group('collision')
            grasps_gp.create_group('no_ik')
            grasps_gp.create_group('no_collision')

        # Get the descriptor for the grasp group
        grasp_group = grasp_trials[object_name]['grasps'][grasp_class]

        # return either collision group or no-collision
        return grasp_group

    ###################################
    ######  PART IV : Services ########
    ###################################
    def handle_record_grasp_trial_data(self, req):
        self.curr_object_name = req.object_name

        # r+ : Read/write, file must exist
        with h5py.File(self.grasp_data_file_name, 'r+') as grasp_file:
            # Update grasp meta information and update sess metadata
            self.update_all_metadata(grasp_file, req.is_top_grasp, req.grasp_success_label,
                                     req.collision_to_approach_pose, req.collision_to_grasp_pose,
                                     False,False,False,False)

            # Get a descriptor for grasp trials
            grasp_trials = grasp_file['recording_sessions'][self.curr_sess_name]['grasp_trials']

            # Create new grasp for this group
            no_collision_gp = self.get_grasp_group(grasp_trials, 'no_collision')

            # Create dataset for new grasp
            self.no_collision_id += 1
            nc_id_str = str(self.no_collision_id).zfill(4)
            grasp_group = no_collision_gp.create_group('grasp_' + nc_id_str)

            #Add all the grasp-specific data and store in datasets under new group
            # object_name
            grasp_group.create_dataset('object_name', data=req.object_name)

            # time stamp
            grasp_group.create_dataset('time_stamp', data=req.time_stamp)
            # is_top_grasp
            grasp_group.create_dataset('is_top_grasp', data=req.is_top_grasp)
            # grasp_success_label
            grasp_group.create_dataset('grasp_success_label', data=req.grasp_success_label)
            # collision label
            grasp_group.create_dataset('collision_to_approach_pose',
                                       data=req.collision_to_approach_pose)
            grasp_group.create_dataset('collision_to_grasp_pose', data=req.collision_to_grasp_pose)

            # object_world_sim_pose
            grasp_group.create_dataset('object_mesh_frame_world',
                                       data=self.convert_pose_to_list(req.object_mesh_frame_world))

            # desired_preshape_palm_world_pose
            grasp_group.create_dataset('desired_preshape_palm_mesh_frame',
                                       data=self.convert_pose_to_list(
                                           req.desired_preshape_palm_mesh_frame))
            # true_preshape_palm_world_pose
            grasp_group.create_dataset('true_preshape_palm_mesh_frame',
                                       data=self.convert_pose_to_list(
                                           req.true_preshape_palm_mesh_frame))

            # desired_preshape_hithand_joint_state
            grasp_group.create_dataset('desired_preshape_joint_state',
                                       data=req.desired_joint_state.position)
            # true_preshape_hithand_joint_state
            grasp_group.create_dataset('true_preshape_joint_state',
                                       data=req.true_joint_state.position)
            # closed_hithand_joint_state
            grasp_group.create_dataset('closed_joint_state', data=req.closed_joint_state.position)
            # lifted_hithand_joint_state
            grasp_group.create_dataset('lifted_joint_state', data=req.lifted_joint_state.position)

        # Return response, context manager closes file
        res = RecordGraspTrialDataResponse(success=True)
        return res

    def handle_record_grasp_trial_multi_obj_data(self, req):
        self.curr_object_name = req.object_name

        # r+ : Read/write, file must exist
        with h5py.File(self.grasp_data_file_name, 'r+') as grasp_file:
            # Update grasp meta information and update sess metadata
            self.update_all_metadata(grasp_file, req.is_top_grasp, req.grasp_success_label,
                                     False, False,
                                     req.grasp_pose_collide_target_object,
                                     req.grasp_pose_collide_obstacle_objects,
                                     req.close_finger_collide_obstacle_objects,
                                     req.lift_motion_moved_obstacle_objects)
            
            # Get a descriptor for grasp trials
            grasp_trials = grasp_file['recording_sessions'][self.curr_sess_name]['grasp_trials']

            # Create new grasp for this group
            no_collision_gp = self.get_grasp_group(grasp_trials, 'no_collision')

            # Create dataset for new grasp
            self.no_collision_id += 1
            nc_id_str = str(self.no_collision_id).zfill(4)
            grasp_group = no_collision_gp.create_group('grasp_' + nc_id_str)

            # Add all the grasp-specific data and store in datasets under new group
            # object_name
            grasp_group.create_dataset('object_name', data=req.object_name)
            grasp_group.create_dataset('obstacle1_name', data=req.obstacle1_name)
            grasp_group.create_dataset('obstacle2_name', data=req.obstacle2_name)
            grasp_group.create_dataset('obstacle3_name', data=req.obstacle3_name)

            # time stamp
            grasp_group.create_dataset('time_stamp', data=req.time_stamp)
            # is_top_grasp
            grasp_group.create_dataset('is_top_grasp', data=req.is_top_grasp)
            # grasp_success_label
            grasp_group.create_dataset('grasp_success_label', data=req.grasp_success_label)
            # collision label
            grasp_group.create_dataset('collision_to_approach_pose',
                                       data=req.collision_to_approach_pose)
            grasp_group.create_dataset('collision_to_grasp_pose', data=req.collision_to_grasp_pose)
            # label for multi obj data generation
            grasp_group.create_dataset('grasp_pose_collide_target_object',
                                       data=req.grasp_pose_collide_target_object)
            grasp_group.create_dataset('grasp_pose_collide_obstacle_objects',
                                       data=req.grasp_pose_collide_obstacle_objects)
            grasp_group.create_dataset('close_finger_collide_obstacle_objects',
                                       data=req.close_finger_collide_obstacle_objects)
            grasp_group.create_dataset('lift_motion_moved_obstacle_objects', 
                                       data=req.lift_motion_moved_obstacle_objects)
            
            # object_world_sim_pose
            grasp_group.create_dataset('object_mesh_frame_world',
                                       data=self.convert_pose_to_list(req.object_mesh_frame_world))
            grasp_group.create_dataset('obstacle1_mesh_frame_world',
                                       data=self.convert_pose_to_list(req.obstacle1_mesh_frame_world))
            grasp_group.create_dataset('obstacle2_mesh_frame_world',
                                       data=self.convert_pose_to_list(req.obstacle2_mesh_frame_world))
            grasp_group.create_dataset('obstacle3_mesh_frame_world',
                                       data=self.convert_pose_to_list(req.obstacle3_mesh_frame_world))

            # desired_preshape_palm_world_pose
            grasp_group.create_dataset('desired_preshape_palm_mesh_frame',
                                       data=self.convert_pose_to_list(
                                           req.desired_preshape_palm_mesh_frame))
            # true_preshape_palm_world_pose
            grasp_group.create_dataset('true_preshape_palm_mesh_frame',
                                       data=self.convert_pose_to_list(
                                           req.true_preshape_palm_mesh_frame))

            #desired_preshape_hithand_joint_state
            grasp_group.create_dataset('desired_preshape_joint_state',
                                       data=req.desired_joint_state.position)
            #true_preshape_hithand_joint_state
            grasp_group.create_dataset('true_preshape_joint_state',
                                       data=req.true_joint_state.position)
            #closed_hithand_joint_state
            grasp_group.create_dataset('closed_joint_state', data=req.closed_joint_state.position)
            #lifted_hithand_joint_state
            grasp_group.create_dataset('lifted_joint_state', data=req.lifted_joint_state.position)

        #Return response, context manager closes file
        res = RecordGraspTrialMultiObjDataResponse(success=True)
        return res

    def handle_record_collision_data(self, req):
        """ This records all the collision grasp data.
        """
        self.curr_object_name = req.object_name

        # r+ : Read/write, file must exist
        with h5py.File(self.grasp_data_file_name, 'r+') as grasp_file:
            # get a descriptor for grasp trials
            grasp_trials = grasp_file['recording_sessions'][self.curr_sess_name]['grasp_trials']

            # Verify if object name already exists, if not create first
            collision_gp = self.get_grasp_group(grasp_trials, grasp_class=req.failure_type)

            # Update all metadata
            self.update_all_metadata_collision(grasp_file,
                                               num_grasps=len(req.preshapes_palm_mesh_frame_poses))

            # Iterate through all grasps
            for obj, palm, joints in zip(req.object_world_poses,
                                         req.preshapes_palm_mesh_frame_poses,
                                         req.preshape_hithand_joint_states):
                # Convert poses to lists
                obj_l = self.convert_pose_to_list(obj)
                palm_l = self.convert_pose_to_list(palm)

                # Increase collision id counter
                self.collision_id[req.failure_type] += 1
                collision_id_str = req.failure_type + '_' + str(self.collision_id[req.failure_type]).zfill(4)

                # Create new group under collision group for each grasp attempt
                grasp_gp = collision_gp.create_group(collision_id_str)

                # Object in world
                grasp_gp.create_dataset('object_mesh_frame_world', data=obj_l)
                # Palm pose w.r.t to object mesh frame
                grasp_gp.create_dataset('desired_palm_pose_mesh_frame', data=palm_l)
                # Desired joint state
                grasp_gp.create_dataset('desired_joint_state', data=joints.position)

        # Return response
        res = RecordCollisionDataResponse(success=True)
        return res

    def create_record_grasp_trial_data_server(self):
        rospy.Service('record_grasp_trial_data', RecordGraspTrialData,
                      self.handle_record_grasp_trial_data)
        rospy.loginfo('Service record_grasp_trial_data:')
        rospy.loginfo('Ready to record your awesome grasp data.')
        
    def create_record_grasp_trial_multi_obj_data_server(self):
        rospy.Service('record_grasp_trial_multi_obj_data', RecordGraspTrialMultiObjData,
                      self.handle_record_grasp_trial_multi_obj_data)
        rospy.loginfo('Service record_grasp_trial_multi_obj_data:')
        rospy.loginfo('Ready to record your awesome grasp data.')
        
    def create_record_collision_data_server(self):
        rospy.Service('record_collision_data', RecordCollisionData,
                      self.handle_record_collision_data)
        rospy.loginfo('Service record_collision_data:')
        rospy.loginfo('Ready to record your awesome collision data.')


DEBUG = False

if __name__ == '__main__':
    rgd = RecordGraspData()
    rgd.create_record_grasp_trial_data_server()
    rgd.create_record_collision_data_server()
    rgd.create_record_grasp_trial_multi_obj_data_server()
    if DEBUG:
        req = RecordCollisionDataRequest(object_name='kit_baking')
        req.object_world_poses = [PoseStamped()]
        req.preshape_hithand_joint_states = [JointState()]
        req.preshapes_palm_mesh_frame_poses = [PoseStamped()]
        rgd.handle_record_collision_data(req)

        req = RecordGraspTrialDataRequest(object_name='kit_baking')
        req.time_stamp = 'lol'
        req.is_top_grasp = True
        req.grasp_success_label = 1

        req.collision_to_approach_pose = 1
        req.collision_to_grasp_pose = 1

        # The true spawn pose in world frame
        req.object_mesh_frame_world = [PoseStamped()]
        req.desired_preshape_palm_mesh_frame = [PoseStamped()]
        req.true_preshape_palm_mesh_frame = [PoseStamped()]
        # Hithand jointstates
        req.desired_joint_state = [JointState()]
        req.true_joint_state = [JointState()]
        req.closed_joint_state = [JointState()]
        req.lifted_joint_state = [JointState()]
        rgd.handle_record_grasp_trial_data(req)

    rospy.spin()
