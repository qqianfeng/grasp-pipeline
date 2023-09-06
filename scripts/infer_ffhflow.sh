#!/bin/bash
xterm -hold -e 'pwd; cd $(rosparam get hithand_ws_path)/src/grasp-pipeline/scripts; source /home/$USER/miniconda3/bin/activate prohmr; python3 infer_ffhflow.py'
