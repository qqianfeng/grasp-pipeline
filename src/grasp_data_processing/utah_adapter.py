""" The script takes as input the generated grasp data and generates same amount of Utah format training data as in 
https://arxiv.org/pdf/2001.09242.pdf

Side:     +1559 -2616
Overhead: +749  -4064
"""
from ..grasp_client.grasp_sim_client import GraspClient

grasp_client = GraspClient(grasp_data_recording_path='')