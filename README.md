# Grasp-pipeline

## Using the real camera.
### Preparation
1. Perform camera calibration
2. Currently camera transform hardcoded in `segment_object_server.py`. Make changes here.
    - Note: Camera coordinate system in and real not equal.
    - Adjust the workspace boundaries in x_min, x_max, y_min, y_max

### Sample grasps
1. Execute the script `src/grasp_pipeline/main/real/eval_vae`