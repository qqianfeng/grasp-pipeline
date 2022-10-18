import sys
import os
import time
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math

# Add GraspInference to the path and import
sys.path.append(os.path.join(os.environ['GRASP_PIPELINE_PATH'], 'src'))
from graspInference import *

# Import robot controller
from ar_toolkit.robots import Diana7Robot

# Create skew-symmetric matrix from vector
def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

# Create rotation matrix around z-axis
def rot_z(z):
    return np.array([[math.cos(z), -math.sin(z), 0],
                     [math.sin(z), math.cos(z), 0],
                     [0, 0, 1]])

# Transformations
hand_R_ee = np.zeros((3, 3))
hand_R_ee[0,2] = -1
hand_R_ee[1,0] = 1
hand_R_ee[2,1] = -1

cam_R_ee = np.zeros((3, 3))
cam_R_ee[0,1] = -1
cam_R_ee[1,0] = -1
cam_R_ee[2,2] = -1
cam_t_ee = np.array([0.1, 0, 0])

# Move robot to home position
robot = Diana7Robot("diana")
robot.move_ptp([0,0,0,1.57079632679,0,-1.57079632679,0])

# Initialize variables
n_poses = 1000
grasp_execution_threshold = 0.8
k_p = [0.1, 0.05]
v_des_max = [1.0, 1.0]
start = time.process_time()
time_abs = 0.0
robot_pos = np.zeros(3)
robot_rot = np.zeros(3)

# Construct class for FFHNet inference
gi = GraspInference()

# Initialize lists for plot
palm_pos_list = []
palm_rot_list = []
p_max_list = []
p_best_list = []
norm_pos_diff_list = []
norm_rot_diff_list = []
current_success_list = []
center_pos_list = []
c_list = []

# Get current pose of the robot
robot_pos_init = robot.get_cartesian_pose()[:3, 3:4].flatten()

i = 0
try:
    while i < 1000:

        # Run inference on FFHNet
        if int(os.environ['VISUALIZE']):
            grasp_dict, _ = gi.handle_infer_grasp_poses(n_poses)

            palm_poses, joint_confs = gi.handle_evaluate_and_filter_grasp_poses(
                grasp_dict, thresh=0.5)

        else:
            grasp_dict, center_transf = gi.handle_infer_grasp_poses(n_poses)
            p_success = gi.handle_evaluate_grasp_poses(grasp_dict)

        # Rotate grasp poses into end effector frame
        grasp_dict_rot = grasp_dict['rot_matrix'].detach().cpu().numpy() @ hand_R_ee.T

        # Save rotation and translation of grasp poses
        grasp_rot = R.from_matrix(grasp_dict_rot)
        grasp_pos = grasp_dict['transl'].detach().cpu().numpy() @ cam_R_ee.T
        center_translation = (center_transf[:, :3] @ cam_R_ee.T)[0]

        # Update rotation and translation
        try:
            cart_pose = robot.get_cartesian_pose()
        except Exception as e:
            print(e)
            break
        robot_pos = cart_pose[:3, 3:4].flatten() - robot_pos_init
        robot_rot_mat = cart_pose[:3, :3]
        robot_rot = R.from_matrix(robot_rot_mat).as_rotvec()

        # Update grasp poses (simulation, in reality point clouds are updated)
        grasp_pos = grasp_pos - robot_pos
        grasp_rot = R.from_rotvec(grasp_rot.as_rotvec() - robot_rot)
        center_translation = center_translation @ robot_rot_mat
        center_translation -= robot_pos
        center_pos_list.append(copy.deepcopy(center_translation))
        
        # Calculate norm of translational and roational offsets
        diff_pos = np.linalg.norm(grasp_pos, 1, axis=1)
        diff_rot = np.linalg.norm(grasp_rot.as_rotvec(), 1, axis=1) #abs(grasp_rot.as_rotvec()[:, 2])

        # Find best grasp
        p_best = np.argmax(p_success - 1*diff_pos - 1*diff_rot)

        # Set orientation towards center of point clound (align -z of EE with vector to center)
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        center_translation /= np.linalg.norm(center_translation)
        v_hat = skew(np.array([center_translation[1], -center_translation[0], 0]))
        rot_align_z_mat = np.identity(3) + v_hat - center_translation[2] * v_hat @ v_hat

        # Rotate around z to to align grasp pose
        rot_align_z = R.from_matrix(rot_z(grasp_rot.as_rotvec()[p_best, 2]) @ rot_align_z_mat).as_rotvec()

        # Metric to banalnce aligning z and grasp rotation
        c_bal = np.clip(10*diff_pos[p_best], 0.0, 1.0)
        rot_des = c_bal * rot_align_z + (1-c_bal) * grasp_rot.as_rotvec()[p_best]
        c_list.append(c_bal.copy())

        # Append results to list (for plot)
        norm_pos_diff_list.append(diff_pos[p_best])
        norm_rot_diff_list.append(diff_rot[p_best])
        palm_pos_list.append(grasp_pos[p_best])
        palm_rot_list.append(rot_des)
        p_best_list.append(p_success[p_best].copy())

        # Get max probability of all grasps
        p_argmax = np.argmax(p_success)
        p_max_list.append(p_success[p_argmax].copy())

        # Move robot towards grasp
        dx_update = k_p[0] * grasp_pos[p_best]

        dr_axis_update = rot_des
        dr_axis_update /= np.linalg.norm(dr_axis_update)
        dr_angle_update = k_p[1] * np.linalg.norm(rot_des)

        # Limit linear and angular velocity
        if np.linalg.norm(dx_update) > v_des_max[0]:
            dx_update /= np.linalg.norm(dx_update) * v_des_max[0]
        dr_angle_update = np.clip(dr_angle_update, -v_des_max[1], v_des_max[1])

        try:
            robot.move_linear_relative((dx_update, (dr_axis_update, dr_angle_update)))
        except Exception as e:
            print(e)
            break

        # Check probability of current configuration to succeed
        grasp_current = dict()
        grasp_current['transl'] = torch.tensor(robot_pos.reshape((1, -1)) @ cam_R_ee).float().cuda()
        grasp_current['rot_matrix'] = torch.tensor(robot_rot_mat.reshape((1, 3, 3)) @ hand_R_ee).float().cuda()
        grasp_current['joint_conf'] = grasp_dict['joint_conf'][p_best]
        grasp_current['z'] = grasp_dict['z'][p_best]
        grasp_current['bps_object'] = grasp_dict['bps_object'][p_best]
        current_success = gi.handle_evaluate_grasp_poses(grasp_current)
        print("Current success: ", current_success)
        current_success_list.append(current_success.copy())

        # Time analysis
        i += 1
        time_loop = time.process_time() - start - time_abs
        print("Time of last cycle: ", time_loop)
        time_abs += time_loop

        # Execute grasp if probability is higher than threshold
        if current_success > grasp_execution_threshold:
            print("\n####################")
            print("# Grasp Execution! #")
            print("####################\n")
            break

except KeyboardInterrupt:
    pass

print("Overall time: ", time_abs)
print("Avg time per cicle: ", time_abs/i, " | Cicles per second: ", i/time_abs)

# Plot result
#plt.plot(palm_pos_list, label=["x", "y", "z"])
#plt.plot(palm_rot_list, label=["rx", "ry", "rz"])
#plt.plot(p_max_list, label="p_max")
# plt.plot(p_best_list, label="p_best")
plt.plot(norm_pos_diff_list, label="delta_x")
plt.plot(norm_rot_diff_list, label="delta_rot")
seq_len = len(current_success_list)
plt.plot(grasp_execution_threshold*np.ones(seq_len), label="grasp_execution_threshold")
plt.plot(current_success_list, label="exp_success")
# plt.plot(center_pos_list, label="center_pos")
plt.plot(c_list, label="c_bal")
plt.legend(loc="upper left")
plt.show()
