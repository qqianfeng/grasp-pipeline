import sys
import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

# Add GraspInference to the path and import
sys.path.append(os.path.join(os.environ['GRASP_PIPELINE_PATH'], 'src'))
from graspInference import *

n_poses = 1000
gi = GraspInference()

start = time.process_time()
time_abs = 0.0
palm_poses_list = []
p_max_list = []

i = 0
while i < 1000:
    if int(os.environ['VISUALIZE']):
        palm_poses_obj_frame, joint_confs = gi.handle_infer_grasp_poses(n_poses)

        if 0:
            for p in palm_poses_obj_frame:
                print("palm_poses_obj_frame", p)

            for j in joint_confs:
                print("joint_confs", j)

        palm_poses, joint_confs = gi.handle_evaluate_and_filter_grasp_poses(
            palm_poses_obj_frame, joint_confs, thresh=0.9)

    else:
        grasp_dict = gi.handle_infer_grasp_poses(n_poses)
        p_success = gi.handle_evaluate_grasp_poses(grasp_dict)

    # Calculate difference to last palm pose
    #if i > 0:
    #    print("Relative offset of best pose: ", palm_poses_list[i-1]-palm_poses[0].pose)

    # Get max p
    p_argmax = np.argmax(p_success)
    p_max_list.append(p_success[p_argmax].copy())
    palm_poses_list.append(grasp_dict['transl'][p_argmax].detach().cpu().numpy())

    i += 1

    time_loop = time.process_time() - start - time_abs
    print("Time of last cycle: ", time_loop)
    time_abs += time_loop

print("Overall time: ", time_abs)
print("Avg time per cicle: ", time_abs/i, " | Cicles per second: ", i/time_abs)

plt.plot(palm_poses_list, label=["x", "y", "z"])
plt.plot(p_max_list, label="p")
plt.legend(loc="upper left")
plt.show()
