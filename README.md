# Grasp Pipeline

The grasp pipeline houses all the grasping related services and scripts to generate ground truth data and evaluate the machine learning model (FFHNet). It offers functionality for saving grasping data, sampling grasps, executing grasps, planning trajectories, spawning and removing objects from simulation and much more.

## Installation

Create virtual environment with Python 3:
```
conda create -n grasp-pipeline python=3.9
```

Install requirements with pip:
```
pip install -r requirements.txt
```

Install pytorch:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

## To run in real world:

Set environment variables:
```
source grasp_env.txt
```

Activate virtual environment:
```
conda activate grasp-pipeline
```

Run grasp pipeline:
```
python scripts/grasp_pipeline_real.py
```
