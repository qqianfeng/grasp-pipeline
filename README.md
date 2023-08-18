# Grasp Pipeline

The grasp pipeline houses all the grasping related services and scripts to generate ground truth data and evaluate the machine learning model (FFHNet). It offers functionality for saving grasping data, sampling grasps, executing grasps, planning trajectories, spawning and removing objects from simulation and much more.

## Installation

### Running Inference: Python 2 vs. Python 3

In order to run inference with the FFHNet it is necessary to install miniconda with python 3.8 and xterm. Inside of miniconda
the (base) environment must hold the pip-installed packages listed in `misc/conda_base_requirements.txt`.
The script `scripts/encode_pcd_with_bps.sh` can be started from a ros launch file, creates an additional xterm terminal, sources
the conda python 3.8 (base) environment and starts the server `encode_pcd_with_bps.py` from there. The server uses roslibpy to handle communication between ROS/python 2.7 and python 3.8.

### VSCODE debugger for python2

In order to debug the python2 script in vscode, you have to downgrade python extension to an old version in this link:
<https://stackoverflow.com/questions/72214043/how-to-debug-python-2-7-code-with-vs-code>

## Difference between hithand and robotiq pipeline
In repo grasp-pipeline, switch branch between `collision-data-generation` for hithand or `robotiq` for robotiq.

In repo panda-simulator, switch branch between
`melodic-hithand` for hithand or `agile-modifications` for robotiq.

## Run test for robotiq

Start the panda_simulator with robotiq\
`roslaunch panda_gazebo panda_robotiq3f.launch`

Start the panda_robotiq3f_moveit\
`roslaunch robotiq_3f_rviz_moveit_panda panda_robotiq3f_moveit.launch`

Start the grasp_pipeline. This exposes the the robotiq grasping servers.\
`roslaunch grasp_pipeline grasp_pipeline_servers_robotiq.launch`

Open another terminal and activate conda base environment\
`cd hithand_ws/src/grasp-pipeline/tests`\
`python2 test_grasping_robotiq.py`

## To run data generation of single object in simulation with robotiq gripper

Start the panda_simulator with robotiq\
`roslaunch panda_gazebo panda_robotiq3f.launch`

Start the panda_robotiq3f_moveit\
`roslaunch robotiq_3f_rviz_moveit_panda panda_robotiq3f_moveit.launch`

Start the grasp_pipeline. This exposes the the robotiq grasping servers.\
`roslaunch grasp_pipeline grasp_pipeline_servers_robotiq.launch`

Open another terminal and run the data generation \
(Note that the metadata_handler.py and object_names_in_datasets.py scripts should be modified to select the datasets.)\
`cd ~/path/to/grasp-pipeline/src/grasp_pipeline/main/sim`\
`python2 data_gen_pipeline_robotiq.py`

## Run evaluation of model

Start the panda_simulator\
`roslaunch panda_gazebo panda_hithand.launch`

Start the panda_hithand_moveit_config\
`roslaunch panda_hithand_moveit_config panda_hithand_moveit.launch`

### To run data generation of single object in simulation

Start the grasp_pipeline. This exposes the the grasping servers.\
`roslaunch grasp_pipeline grasp_pipeline_servers.launch`

Open another terminal and activate conda base environment\
`cd ~/path/to/grasp-pipeline/src/grasp_pipeline/main/sim`\
`python2 data_gen_pipeline.py`

### To run evaluation of single object in simulation

Start the grasp_pipeline. This exposes the the grasping servers.\
`roslaunch grasp_pipeline grasp_pipeline_servers_eval_ffhnet.launch`

Open another terminal and activate conda base environment\
`cd ~/path/to/grasp-pipeline/src/grasp_pipeline/main/sim`\
`python2 eval_ffhgen_sim.py`

### Data post processing after data generation
see FFHNet repo readme. https://github.com/qianbot/FFHNet-dev

## Grasp Data Preparation

This section details the step to bring the grasping data into the right format.

1. Collect all ground truth data into one directory with folders for each recording.

```bash
    Data
      ├── 2021-03-02
      |         ├── grasp_data (folder with images of grasp)
      |         ├── grasp_data.h5 (file with all the poses)
      ...
      ├── 2021-03-10
```

2. Execute the script `grasp_pipeline/src/grasp_pipeline/grasp_data_processing/merge_raw_grasp_data.py` \
This will go through all the folders under `Data/` and combine all `grasp_data.h5` files into one combined `grasp_data_all.h5`.

3. Execute the script `grasp_pipeline/src/grasp_pipeline/grasp_data_processing/data_augmentation.py`
This will go through all the objects in `grasp_data_all.h5` and spawn each object in `n_pcds_per_object` random positions and orientations, record a segmented point cloud observation as well as the transformation between the mesh frame of the object and the object centroid frame. All transforms get stored in `pcd_transforms.h5`.
3.1 The script also creates `metadata.csv` which contains the columns \
object_name | collision | negative | positive | train | test | val \
An `X` in train/test/val indicates that this object belongs to the training, test or val set.
A `XXX` indicates that the object should be excluded, because the data acquisition was invalid.

4. Execute the script `ffhnet/scripts/train_test_val_split.py` which given the `metadata.csv` file splits the data in the three folders `train`, `test`, `val`. Under each of the three folders lies a folder `point_clouds` with N `.pcd` files of objects from different angles.

5. Execute the script `bps_torch/convert_pcds_to_bps.py` which will first compute a BPS (basis point set) and then computes the bps representation for each object storing them in a folder `bps` under the respective object name.

### To run in real world with calibrated camera

`roslaunch grasp_pipeline grasp_pipeline_servers_real_vision_only.launch`

Currently camera transform hardcoded in `segment_object_server.py`. Make changes here.
    - Note: Camera coordinate system in and real not equal.
    - Adjust the workspace boundaries in x_min, x_max, y_min, y_max

Execute the script:\
`cd ~/path/to//grasp-pipeline/src/grasp_pipeline/main/real`\
`python2 eval_ffhgen_real.py`

## Issue

Current segmentation method use plane segmentation. Every point cloud except the estimated plane will be treated as objects for bbox calculation. Removing noises from background is neccesary.
