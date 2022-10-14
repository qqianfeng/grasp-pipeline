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

Activate conda environment:
```
conda activate grasp-pipeline
```

Start shell environment:
```
eval $(arpm_env ar-toolkit/9.0.0@ar/stable)
```

Add DDS message path to env:
```
export AR_IDL_DIRS=$AR_IDL_DIRS:$PWD/idl
```

Exceute grasp pipeline:
```
python scripts/grasp_pipeline_real.py
```
