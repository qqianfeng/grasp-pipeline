#!/bin/bash
xterm -hold -e 'pwd; cd $(rosparam get hithand_ws_path)/src/grasp-pipeline/scripts; source /home/$USER/miniconda3/bin/activate; python3 encode_pcd_with_bps.py'
