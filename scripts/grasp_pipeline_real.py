import sys
import os
import time

# Add GraspInference to the path and import
sys.path.append(os.path.join(os.environ['GRASP_PIPELINE_PATH'], 'src'))
from graspInference import *

n_poses = 100
gi = GraspInference()

for i in range(1):
    palm_poses_obj_frame, joint_confs = gi.handle_infer_grasp_poses(n_poses)

    for p in palm_poses_obj_frame:
        print("palm_poses_obj_frame", p)

    for j in joint_confs:
        print("joint_confs", j)

    palm_poses, joint_confs = gi.handle_evaluate_and_filter_grasp_poses(
        palm_poses_obj_frame, joint_confs, thresh=0.9)
