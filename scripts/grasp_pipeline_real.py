import sys
import os
import time
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Add GraspInference to the path and import
sys.path.append(os.path.join(os.environ['GRASP_PIPELINE_PATH'], 'src'))
from graspInference import *

# Import robot controller
from ar_toolkit.robots import Diana7Robot

# Transformations
ee_R_hand = np.zeros((3, 3))
ee_R_hand[0,2] = -1
ee_R_hand[1,0] = -1
ee_R_hand[2,1] = 1

cam_R_ee = np.zeros((3, 3))
cam_R_ee[0,1] = -1
cam_R_ee[1,0] = -1
cam_R_ee[2,2] = -1
cam_t_ee = np.array([0.1, 0, 0])

base_R_ee= np.identity(3)
base_R_ee[1,1] = -1
base_R_ee[2,2] = -1

# Move robot to home position
robot = Diana7Robot("diana")
robot.move_ptp([0,0,0,1.57079632679,0,-1.57079632679,0])

n_poses = 1000
gi = GraspInference()

start = time.process_time()
time_abs = 0.0
palm_pos_list = []
palm_rot_list = []
p_max_list = []
p_best_list = []
norm_pos_diff_list = []
norm_rot_diff_list = []

x = np.zeros(3)
r = np.zeros(3)

x_init = robot.get_cartesian_pose()

i = 0
while i < 100:
    if int(os.environ['VISUALIZE']):
        grasp_dict = gi.handle_infer_grasp_poses(n_poses)

        palm_poses, joint_confs = gi.handle_evaluate_and_filter_grasp_poses(
            grasp_dict, thresh=0.5)

    else:
        grasp_dict = gi.handle_infer_grasp_poses(n_poses)
        p_success = gi.handle_evaluate_grasp_poses(grasp_dict)

    # Calculate difference to last palm pose
    #if i > 0:
    #    print("Relative offset of best pose: ", palm_poses_list[i-1]-palm_poses[0].pose)

    # Find pose closest to previous pose
    #rot = R.from_matrix(grasp_dict['rot_matrix'].detach().cpu().numpy())
    pos = grasp_dict['transl'].detach().cpu().numpy() @ cam_R_ee.T

    # Update rotation and translation
    cart_pose = robot.get_cartesian_pose() - x_init
    x = cart_pose[:3, 3:4].flatten()
    cart_pose_R = ee_R_hand @ cart_pose[:3, :3]
    r = R.from_matrix(cart_pose_R).as_rotvec()

    # Rotate grasps into camera frame
    #cam_R_hand = ee_R_hand @ cart_pose_R @ base_R_ee @ cam_R_base
    #rot = R.from_matrix(cam_R_hand @ rot.as_matrix())

    # Update pose (simulation)
    pos = pos - x
    print(pos)
    #rot = R.from_rotvec(rot.as_rotvec() - r)
    
    diff_pos = np.linalg.norm(pos, 1, axis=1)
    #diff_rot = np.linalg.norm(rot.as_rotvec(), 1, axis=1)
    
    p_best = np.argmax(p_success - 5*diff_pos)# - 0.5*diff_rot)
    norm_pos_diff_list.append(diff_pos[p_best])
    #norm_rot_diff_list.append(diff_rot[p_best])

    palm_pos_list.append(pos[p_best])
    #palm_rot_list.append(rot.as_rotvec()[p_best])
    p_best_list.append(p_success[p_best].copy())

    # Get max p
    p_argmax = np.argmax(p_success)
    p_max_list.append(p_success[p_argmax].copy())
    #palm_poses_list.append(grasp_dict['transl'][p_argmax].detach().cpu().numpy())

    i += 1

    time_loop = time.process_time() - start - time_abs
    print("Time of last cycle: ", time_loop)
    time_abs += time_loop

    # Move robot towards grasp
    dx_update = pos[p_best]

    dr_axis_update = np.array([1,0,0])#rot[p_best].as_rotvec()
    #dr_axis_update /= np.linalg.norm(dr_axis_update)
    dr_angle_update = 0#- np.linalg.norm(rot[p_best].as_rotvec())

    p = 0.1
    try:
        robot.move_linear_relative((p*dx_update, (dr_axis_update, p*dr_angle_update))) 
    except Exception as e:
        print(e)
        break

print("Overall time: ", time_abs)
print("Avg time per cicle: ", time_abs/i, " | Cicles per second: ", i/time_abs)

plt.plot(palm_pos_list, label=["x", "y", "z"])
#plt.plot(palm_rot_list, label=["rx", "ry", "rz"])
plt.plot(p_max_list, label="p_max")
plt.plot(p_best_list, label="p_best")
plt.plot(norm_pos_diff_list, label="delta_x")
#plt.plot(norm_rot_diff_list, label="delta_rot")
plt.legend(loc="upper left")
plt.show()
