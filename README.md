# Grasp Pipeline

The grasp pipeline houses all the grasping related services and scripts to generate ground truth data and evaluate the machine learning model (FFHNet). It offers functionality for saving grasping data, sampling grasps, executing grasps, planning trajectories, spawning and removing objects from simulation and much more.

## Installation

### Running Inference: Python 2 vs. Python 3

In order to run inference with the FFHNet it is necessary to install miniconda with python 3.8 and xterm. Inside of miniconda
the (base) environment must hold the pip-installed packages listed in `misc/conda_base_requirements.txt`.
The script `scripts/encode_pcd_with_bps.sh` can be started from a ros launch file, creates an additional xterm terminal, sources
the conda python 3.8 (base) environment and starts the server `encode_pcd_with_bps.py` from there. The server uses roslibpy to handle communication between ROS/python 2.7 and python 3.8.

## Run evaluation of model

Start the panda_simulator\
`roslaunch panda_gazebo panda_hithand.launch`

Start the panda_hithand_moveit_config\
`roslaunch panda_hithand_moveit_config panda_hithand_moveit.launch`

1. To run in simulation
Start the grasp_pipeline. This exposes the the grasping servers.\
`roslaunch grasp_pipeline grasp_pipeline_servers_eval_ffhnet.launch`

Open another terminal and activate conda base environment\
`cd ~/hand_ws/src/grasp-pipeline/src/grasp_pipeline/main/sim`\
`python2 eval_ffhgen_sim.py`

2. To run in real world
`roslaunch grasp_pipeline grasp_pipeline_servers_real_vision_only.launch`

Tune distance threshold parameter to exclude background\
`~/hand_ws/src/grasp-pipeline/src/grasp_pipeline/server/sim/segment_object_server.py`

`cd ~/hand_ws/src/grasp-pipeline/src/grasp_pipeline/main/real`\
`python2 eval_ffhgen_real.py`

## Issue

Current segmentation method use plane segmentation. Every point cloud except the estimated plane will be treated as objects for bbox calculation. Removing noises from background is neccesary.
