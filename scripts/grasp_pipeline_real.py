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

# DDS imports
import ar_dds as dds

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

# DDS subscriber functions
def interrupt_signal_handler(_signal_number, _frame):
    """SIGINT/SIGTSTP handler for gracefully stopping an application."""
    print("Caught interrupt signal. Stop application!")
    global shutdown_requested
    shutdown_requested = True

# Diana X1 rotation
d7_T_dx1 = np.identity(3)
d7_T_dx1[0,0] = -1
d7_T_dx1[2,2] = -1

def poll_telemetry_data():
    received_data = listener.take(True)
    for sample in received_data:
        if not sample.info.valid:
            continue

        T = sample.data["T_flange2base"]

        pos = T["translation"]
        robot_pos = np.array([pos['x'], pos['y'], pos['z']])

        quat = T["rotation"]
        quat_arr = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
        robot_rot = R.from_quat(quat_arr)
        robot_rot = R.from_matrix(d7_T_dx1 @ robot_rot.as_matrix())

    return robot_pos, robot_rot

# Initialize DDS Domain Participant
participant = dds.DomainParticipant(domain_id=0)

# Create subscriber using the participant
listener = participant.create_subscriber_listener("ar::interfaces::dds::robot::generic::telemetry_v1",
                                                  "diana.telemetry_v1", None)
print("Subscriber is ready. Waiting for data ...")

# Create Domain Participant and a publisher to publish velocity
publisher = participant.create_publisher("ar::frankenstein_legacy_interfaces::dds::robot::diana::linear_speed_servoing_v1",
                                         'des_cart_vel_msg')

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
#robot = Robot("diana")
#robot.move_ptp([0,0,0,1.57079632679,0,-1.57079632679,0])

# Set hyperparameters
n_poses = 1000                      # Number of poses generated by FFHGenerator in every time step
grasp_execution_threshold = 0.8     # Expected success by FFHEvaluator for grasp execution
dist_align_rot = 10                 # Distance to object in cm when also considering rotation of
                                    # grasp (not only align z axis of EE with object direction)
k_p = [0.1, 0.05]                   # Constant control parameters
v_des_max =[0.05, 0.05]             # Limit of linear and angular velocity

# Initialize variables
start = time.process_time()
time_abs = 0.0

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
robot_pos_init, _ = poll_telemetry_data()

i = 0
try:
    while i < 1000000:

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
        robot_pos, robot_rot = poll_telemetry_data()
        robot_pos = robot_pos - robot_pos_init
        robot_rot_mat = robot_rot.as_matrix()
        robot_rot = robot_rot.as_rotvec()

        # Update grasp poses (simulation, in reality point clouds are updated)
        grasp_pos = grasp_pos - robot_pos
        grasp_rot = R.from_rotvec(grasp_rot.as_rotvec() - robot_rot)
        center_translation = center_translation @ robot_rot_mat @ d7_T_dx1.T
        center_translation -= robot_pos
        center_pos_list.append(copy.deepcopy(center_translation))
        
        # Calculate norm of translational and roational offsets
        diff_pos = np.linalg.norm(grasp_pos, 1, axis=1)
        diff_rot = np.linalg.norm(grasp_rot.as_rotvec(), 1, axis=1)

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
        c_bal = np.clip(dist_align_rot*diff_pos[p_best], 0.0, 1.0)
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
        dr_update = k_p[1] * rot_des

        # Limit linear and angular velocity
        if np.linalg.norm(dx_update) > v_des_max[0]:
            dx_update = dx_update / np.linalg.norm(dx_update) * v_des_max[0]
        if np.linalg.norm(dr_update) > v_des_max[1]:
            dr_update = dr_update / np.linalg.norm(dr_update) * v_des_max[1]

        # Only for diana x1 (TODO: remove)
        dx_update = d7_T_dx1 @ dx_update
        #dr_update = d7_T_dx1 @ dr_update

        # Publish desired velocity
        v_des = np.concatenate([dx_update, dr_update])
        publisher.message["dT"] = v_des
        publisher.publish()

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

publisher.message["dT"] = np.zeros(6)
publisher.publish()

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