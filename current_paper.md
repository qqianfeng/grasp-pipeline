
# Journal Paper of FFHNet (End: 30.12.2022)

## Deadlines

- IROS 2023 RAL: 24.02.2022 -> Response in 3 months, In USA.

## Writing Paper

- Introduction (5%)
- Literature Review (50%)
- Methods (5%)
- Experimental results
- Concolution

## Task 1: Segmentation

1. We need semantic/instance segmentation.
or try this: <https://github.com/bsaund/object_segmentation>

Problem of DL segmentation: it only works for known class object.
How did other paper do for this topic? -> It seems we can use tracking method.

2. Backup plan: tracking from yannick? (Siamask)

## Task 2: Collision Detector (End: 31.12.2022)

0. What should collision detector do?
        - Detect grasp poses that are potentially collide with environments (other objects, ground)
        - no need to detect if poses are IK unsolvable
        - no need to detect if poses lead to self-singularity

1. new collision dataset of objects in cluttered (End: 31.12.2022)

- [ ]: new dataset
        design:
        - [x]: choose_next_grasp_objects
        - [x]: spawn_objects
        - [x]: run pipeline to get pose
        - [x]: spawn new objects
        - [x]: grasp
        - [x]: how to label?

        label:
        - [x]: grasp pose collide target_object -> this means either grasp collides with target object or motion planning is bad (can we get rid of bad motion planning)
        - [x]: grasp pose collide other objects ->  this means either grasp collides with other objects or motion planning is bad
        - [x]: close_finger_collide_other_objects -> this could cause by target object moved and pushes other objects
        - [x]: grasp_success
        - [x]: grasp_failure
        - [x]: grasp lift moved other objects -> objects were moved by lift motion or by previous any motion. Not a problem of grasping but motion planning.

        verify dataset:
        - [ ]: all labels are correctly made
        - [x]: save image: RGB and depth image of complete scene.
        - [x]: grasp pose

        overall generation progress:
        - [x]: BIGBIRD  8/25
        - [x]: KIT  0/89

        - [x]: data clean up

-> training set of poses: 1918300, 60% collision and 40% noncollision ??? Something wrong here?

2. train a collision detector

- [x] Verify the FFHEvaluator on single object dataset
- [x] New model for collision detection?
        PointNet?

        evaluation:
                eval_loss/neg_acc: torch.sum(correct * gt_label) / torch.sum(gt_label)
                eval_loss/pos_acc: torch.sum(correct * (1. - gt_label)) / torch.sum(1. - gt_label)
                eval_loss/total_loss_eva: self.bce_weight * self.BCE_loss(pred_success_p, self.FFHEvaluator.gt_label)

                train_loss/bce_loss: self.bce_weight * self.BCE_loss(pred_success_p, self.FFHEvaluator.gt_label)
                train_loss/total_loss_eva: self.bce_weight * self.BCE_loss(pred_success_p, self.FFHEvaluator.gt_label)

        There is something wrong in the training -> overfitting, 
        Solution 1: reduce the model complexity.(less layer)
                - reduce conv1d layers from 5 to 3 and retry the training. parameters drop from 5m to 3m.
                        here the eval loss dones't increase so much, so it helped!
                - reduce one layer before output
                        eval lose decrease a bit, train loss: 1.1 -> 1.3, eval loss: 4.6 -> 3.9
                - how about more dropout?
                with input/feature transform, pointnet from paper gets acc from 87% to 89%.
                

        [x] Tried to use same label ratio, 0.5/0.5 instead of 0.3/0.3/0.4(hard negative). Training loss increases. doesn't help!

        Missing part from PointNet: random data augmentation:
        We uniformly sample 1024 points on mesh faces according to face area and normalize them into a unit sphere. During training we augment the point cloud on-the-fly by randomly rotating the object along the up-axis and jitter the position of each points by a Gaussian noise with zero mean and 0.02 standard deviation.

        Issue: maybe it's not working at all given only the pose of the hand, a full point cloud will only be then sufficient?
        -> check how other people did for collision detection

- [x] New dataloader

- [ ] Train new colldetr model
        - []: issue: why first eval has 0 % neg acc and 100% pos acc. Is there any bug with eval col?
                it seems the output score from the model are the same for any collision data.
                the model is not learning the feature.

        - []: paper "Collision-Aware Target-Driven Object Grasping in Constrained Environments" transform the point cloud to its current grasp frame
        then voxelize it for later to use.

3. Task 3: Integrate grasp planning pipeline
        segmentation -> list of point cloud groups
        find the object with highest depth average, feed into FFHNet, get list of grasp poses
        grasp poses filtered by collision detector, IK
        execute grasp pose.

- How about motion planning for real world experiment???
middle pose -> grasp pose

<!-- 4. Implement and compare Baseline: Moveit

- [ ]: Is Moveit a valid comparison? Does Moveit supports pointcloud -> voxel for collision checking?
        understand how moveit works (based on FCL or Bullet)
- [ ]: implement Moveit? -->

<!-- ## Better FFHNet

- [x]: dataloader of point cloud directly
- [ ]: data processing:
        how to best downsample the point cloud? voxel_down_sample/uniform_down_sample
- [ ]: try Transformer -->

## Experiments

1. Simulation

single object:

- [x]: Heuristics-based
- [x]: FFHNet: ffhnet_sim_eval_kit_results.ods
- [ ]: DexFFHNet

objects in clutter:

- [ ]:Heuristics-based
        the way of data generation -> load unknown objects from YCB dataset
- [ ]:FFHNet ???
        spawn multiple objects -> grabcut to get mask -> FFHNet -> grasp poses -> spawn other object, execute grasps.
- [ ]:DexFFHNet
        spawn multiple objects -> grabcut to get mask -> FFHNEt -> grasp poses -> collision dector filter poses -> execute grasps.

2. Real world (Yannick)

2.1 single object:

choose 10 from YCB, each give 20 trials and record success rate.
Better to both include top grasp and side grasp. Mark them with success rate separately. Use another realsense or cellphone to record the experiment.

Get grasp pose from model
check if ik solution exists
if yes then compute via pose (extend grasp pose from object center a bit? Also check if via pose has ik solution)
command robot to grasp pose
close finger
lift 10-15cm
record grasp success/failure, FFHEvaluator output score

- [x]:FFHNet
- [ ]:NewFFHNet

2.2 objects in clutter

- []: FFHNet
- []: NewFFHNet

## Issue list

0. (No need to solve?): FFHGenerator generates the grasp distributions highly related to camera frame. If camera pose changes, the grasp distribution changes accordingly.
Solution: randomize camera poses for data generation.

1. (Not important) Problem in eval: All grasps as marked as top grasp?

2. (can be improved with new dataset) Problem with existing dataset. Label for collision grasps. All the grasps with no IK solutions or in collision with environment are labeled as collision for data generation. -> This should be labeled only for grasping with collision.
code: src/grasp_pipeline/server/filter_palm_goal_poses_server.py
