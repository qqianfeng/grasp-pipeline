""" This file is used to evaluate the model and sample grasps. 
"""
import torch
import bps_torch

objs_list = ['']



for obj in objs_list:
    # Spawn model
    
    # Get point cloud (mean-free, orientation of camera frame)

    # Compute BPS of point cloud

    # Build the model

    # Sample N latent variables and get the poses

    # Execute the grasps and record results