# Grasp Pipeline
The grasp pipeline houses all the grasping related services and scripts to generate ground truth data and evaluate the machine learning model (FFHNet). It offers functionality for saving grasping data, sampling grasps, executing grasps, planning trajectories, spawning and removing objects from simulation and much more. 

## Installation
### Running Inference: Python 2 vs. Python 3
In order to run inference with the FFHNet it is necessary to install miniconda with python 3.8 and xterm. Inside of miniconda
the (base) environment must hold the pip-installed packages listed in `misc/conda_base_requirements.txt`. 
The script `scripts/encode_pcd_with_bps.sh` can be started from a ros launch file, creates an additional xterm terminal, sources
the conda python 3.8 (base) environment and starts the server `encode_pcd_with_bps.py` from there. The server uses roslibpy to handle communication between ROS/python 2.7 and python 3.8.