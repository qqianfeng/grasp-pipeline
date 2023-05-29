#!/bin/bash
xterm -hold -e 'pwd; cd $(rosparam get siammask_path)/experiments/siammask_sharp; source /home/$USER/miniconda3/bin/activate; python3 ../../tools/track_obj_and_create_pcd.py --resume SiamMask_DAVIS.pth --config config_davis.json'
