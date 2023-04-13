cmake_minimum_required(VERSION 3.0.2)
project(grasp_pipeline)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  message_runtime
  roscpp
  rospy
  std_msgs
  trajectory_msgs
)

catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_message_files(
  DIRECTORY msg
  FILES
  GraspObject.msg
  ObjectToBeSpawned.msg
)

add_service_files(
    FILES
    BPSEncode.srv
    CheckPoseValidity.srv
    ControlHithand.srv
    DeleteHandGazebo.srv
    EvaluateAndFilterGraspPoses.srv
    ExecuteJointTrajectory.srv
    FilterPalmPoses.srv
    GenVoxelFromPcd.srv
    GraspPreshape.srv
    GraspControl.srv
    InferGraspPoses.srv
    ManageMoveitScene.srv
    MoveArm.srv
    PlanArmTrajectory.srv
    PlanCartesianPathTrajectory.srv
    PlanResetTrajectory.srv
    RecordCollisionData.srv
    RecordGraspDataSim.srv
    RecordGraspTrialData.srv
    RecordGraspTrialMultiObjData.srv
    ResetScene.srv
    SaveVisualData.srv
    SegmentGraspObject.srv
    SimGraspData.srv
    UpdateObjectGazebo.srv
    UpdateHandGazebo.srv
    UpdateObjectPose.srv
    UpdatePalmPose.srv
    VisualizeGraspPoseList.srv
    CreateNewScene.srv
    ClearScene.srv
    ResetScene.srv
    ChangeModelVisibility.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
   geometry_msgs
   trajectory_msgs
)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
    CATKIN_DEPENDS message_runtime
)
