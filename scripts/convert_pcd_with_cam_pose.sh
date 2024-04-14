#!/bin/bash
xterm -hold -e 'pwd; cd $(rosparam get hithand_ws_path)/src/grasp-pipeline/scripts; source /home/$USER/miniconda3/bin/activate ffhnet; python3 convert_pcd_with_cam_pose.py'
