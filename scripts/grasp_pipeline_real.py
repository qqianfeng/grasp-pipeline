import sys
import os
import time

# Add GraspInference to the path and import
sys.path.append(os.path.join(os.environ['HITHAND_WS_PATH'], 'src/grasp-pipeline/src'))
from graspInference import *

n_poses = 10
gi = GraspInference()

for i in range(1):
    palm_poses_obj_frame, joint_confs = gi.handle_infer_grasp_poses(n_poses)
    print("palm_poses_obj_frame", palm_poses_obj_frame)
    print("joint_confs", joint_confs)

    palm_poses, joint_confs = gi.handle_evaluate_and_filter_grasp_poses(
        palm_poses_obj_frame, joint_confs, thresh=0.5)
