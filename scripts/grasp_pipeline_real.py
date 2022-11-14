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

simulation = int(os.environ["SIMULATION"])
use_diana7 = int(os.environ["USE_DIANA7"])

flange_rotation = np.identity(3)
if not use_diana7:
    # Diana X1 flange rotation
    flange_rotation[0,0] = -1
    flange_rotation[2,2] = -1

def poll_telemetry_data():
    received_data = listener.take(True)
    for sample in received_data:
        if not sample.info.valid:
            continue

        T = sample.data[dds_telemetry_topic_name]

        pos = T["translation"]
        robot_pos = np.array([pos['x'], pos['y'], pos['z']])

        quat = T["rotation"]
        quat_arr = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
        robot_rot = R.from_quat(quat_arr)
        robot_rot = R.from_matrix(flange_rotation @ robot_rot.as_matrix())

    return robot_pos, robot_rot

# Initialize DDS Domain Participant
participant = dds.DomainParticipant(domain_id=0)

# Create subscriber using the participant
if use_diana7:
    print("[Info] Using Diana 7.")
    listener = participant.create_subscriber_listener("ar::interfaces::dds::robot::diana::diana7::telemetry_diana_v2",
                                                      "diana7.telemetry_diana_v2", None)
    dds_telemetry_topic_name = "flange_pose"
    print("[Info] Telemetry subscriber is ready. Waiting for data ...")

else:
    print("[Info] Using Diana X1.")
    dds_telemetry_topic_name = "T_flange2base"
    listener = participant.create_subscriber_listener("ar::interfaces::dds::robot::generic::telemetry_v1",
                                                      "diana.telemetry_v1", None)
    print("[Info] Telemetry subscriber is ready. Waiting for data ...")

    # Create Domain Participant and a publisher to publish velocity
    publisher = participant.create_publisher("ar::frankenstein_legacy_interfaces::dds::robot::diana::linear_speed_servoing_v1",
                                            'des_cart_vel_msg')

# Transformations
hand_R_ee = np.zeros((3, 3))
hand_R_ee[0,2] = -1
hand_R_ee[1,1] = -1
hand_R_ee[2,0] = -1

cam_R_ee = np.zeros((3, 3))
cam_R_ee[0,1] = -1
cam_R_ee[1,0] = -1
cam_R_ee[2,2] = -1
cam_t_ee = np.array([0., 0, 0])

# Set hyperparameters
n_poses = 1000                      # Number of poses generated by FFHGenerator in every time step
grasp_execution_threshold = 0.85    # Expected success by FFHEvaluator for grasp execution
dist_align_rot = 10                 # Distance to object in cm when also considering rotation of
                                    # grasp (not only align z axis of EE with object direction)
k_p = [0.2, 0.1]                   # Constant control parameters
k_i = [0.0, 0.0]                    # Integrative control parameters
k_d = [0.,0.]#[0.5, 0.5]                    # Derivative control parameters
v_des_max = [0.05, 0.05]              # Limit of linear and angular velocity

# Initialize variables
time_abs = 0.0
x_update = np.zeros(3)
r_update = np.zeros(3)
dx_update_prev = np.zeros(3)
dr_update_prev = np.zeros(3)

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
grasp_pos_list = []
robot_pos_list = []

# Get current pose of the robot
robot_pos_init, _ = poll_telemetry_data()

# Warmup
grasp_dict, center_transf = gi.handle_infer_grasp_poses(n_poses)
p_success = gi.handle_evaluate_grasp_poses(grasp_dict)

if use_diana7:
    # Start velocity stream with ar-toolkit
    from ar_toolkit.robots import Diana7Robot
    robot = Diana7Robot("diana7")
    robot.start_linear_speed_listener(0.4, 1) # 50 Hz, flange (0: base, 1:flange, 2:tcp)
    robot.linear_speed_servoing(np.zeros(6))
    print("[Info] Started linear velocity stream.")

rand_arr = np.random.rand(6) * np.pi
i = 0
start = time.process_time()
try:
    while i < 1000000:
        # Run inference on FFHNet
        if int(os.environ['VISUALIZE']):
            grasp_dict, _ = gi.handle_infer_grasp_poses(n_poses)

            grasp_dict = gi.handle_evaluate_and_filter_grasp_poses(
                grasp_dict, thresh=0.0)
            exit()

        else:
            grasp_dict, center_transf = gi.handle_infer_grasp_poses(n_poses)
            p_success = gi.handle_evaluate_grasp_poses(grasp_dict)

        if not any(center_transf[0]):
            print("Tracking lost. Stopping robot.")
            break

        # Rotate grasp poses into end effector frame
        grasp_dict_rot = grasp_dict['rot_matrix'].detach().cpu().numpy() @ cam_R_ee.T @ hand_R_ee.T

        # Save rotation and translation of grasp poses
        grasp_rot = R.from_matrix(grasp_dict_rot)
        grasp_pos = grasp_dict['transl'].detach().cpu().numpy() @ cam_R_ee.T + cam_t_ee
        center_translation = (center_transf[:, :3] @ cam_R_ee.T + cam_t_ee)[0]

        # Update rotation and translation
        robot_pos, robot_rot = poll_telemetry_data()
        robot_pos = robot_pos - robot_pos_init
        robot_rot_mat = robot_rot.as_matrix()
        robot_rot = robot_rot.as_rotvec()

        if simulation:
            # Update grasp poses (simulation, in reality point clouds are updated)
            grasp_pos = grasp_pos - robot_pos
            grasp_rot = R.from_rotvec(grasp_rot.as_rotvec() - robot_rot)
            center_translation = center_translation @ robot_rot_mat @ flange_rotation.T
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
        c_bal = np.clip(dist_align_rot*grasp_pos[p_best, 2], 0.0, 1.0)
        rot_des = c_bal * rot_align_z + (1-c_bal) * grasp_rot.as_rotvec()[p_best]
        c_list.append(c_bal.copy())

        # Append results to list (for plot)
        norm_pos_diff_list.append(diff_pos[p_best])
        norm_rot_diff_list.append(diff_rot[p_best])
        palm_pos_list.append(grasp_pos[p_best])
        palm_rot_list.append(rot_des)
        p_best_list.append(p_success[p_best].copy())
        grasp_pos_list.append(grasp_pos[p_best] + robot_pos)
        robot_pos_list.append(robot_pos)

        # Get max probability of all grasps
        p_argmax = np.argmax(p_success)
        p_max_list.append(p_success[p_argmax].copy())

        # Move robot towards grasp
        dx_update = grasp_pos[p_best]
        x_update += dx_update
        ddx_update = (dx_update - dx_update_prev)
        dx_update_prev = dx_update
        dr_update = rot_des
        r_update += dr_update
        ddr_update = (dr_update - dr_update_prev)
        dr_update_prev = dr_update
        lin_update = k_p[0] * dx_update + k_i[0] * x_update + k_d[0] * ddx_update
        rot_update = k_p[1] * dr_update + k_i[1] * r_update + k_d[1] * ddr_update

        # Limit linear and angular velocity
        lin_update = np.clip(lin_update, -v_des_max[0], v_des_max[0])
        rot_update = np.clip(rot_update, -v_des_max[1], v_des_max[1])
        # if np.linalg.norm(lin_update) > v_des_max[0]:
        #     lin_update = lin_update / np.linalg.norm(lin_update) * v_des_max[0]
        # if np.linalg.norm(rot_update) > v_des_max[1]:
        #     rot_update = rot_update / np.linalg.norm(rot_update) * v_des_max[1]

        # Only for diana x1
        lin_update = flange_rotation @ lin_update

        # Publish desired velocity
        v_des = np.concatenate([lin_update, rot_update])
        print(v_des)

        if use_diana7:
            robot.linear_speed_servoing(v_des)
        else:
            publisher.message["dT"] = v_des
            publisher.publish()

        # Check probability of current configuration to succeed
        grasp_current = dict()
        grasp_current['transl'] = torch.tensor((robot_pos.reshape((1, -1)) - cam_t_ee) @ cam_R_ee).float().cuda()
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

# Stop motion
if use_diana7:
    robot.linear_speed_servoing(np.zeros(6))
else:
    publisher.message["dT"] = np.zeros(6)
    publisher.publish()

# Plot result
#plt.plot(palm_pos_list, label=["x", "y", "z"])
#plt.plot(palm_rot_list, label=["rx", "ry", "rz"])
#plt.plot(p_max_list, label="p_max")
# plt.plot(p_best_list, label="p_best")
#plt.plot(norm_pos_diff_list, label="delta_x")
#plt.plot(norm_rot_diff_list, label="delta_rot")
seq_len = len(current_success_list)
plt.plot(grasp_execution_threshold*np.ones(seq_len), label="grasp_execution_threshold")
plt.plot(current_success_list, label="exp_success")
plt.plot(center_pos_list, label="center_pos")
#plt.plot(c_list, label="c_bal")
plt.plot(grasp_pos_list, label=["grasp_x", "grasp_y", "grasp_z"])
plt.plot(robot_pos_list, label=["robot_x", "robot_y", "robot_z"])
plt.legend(loc="upper left")
plt.show()
