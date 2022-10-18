# Grasp Pipeline

## Start gazebo simulator
```
eval $(arpm_env -n ar-gazebo diana_v2-sdf)
gazebo --verbose dianav2.world
```

## Usage
Source environment variables:
```
source grasp_env.txt
```

Exceute grasp pipeline:
```
python scripts/grasp_pipeline_real.py
```
