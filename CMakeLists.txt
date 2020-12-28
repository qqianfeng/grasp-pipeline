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
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  std_msgs
  trajectory_msgs
)

catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_service_files(
    FILES
    ControlHithand.srv
    ExecuteJointTrajectory.srv
    GraspPreshape.srv
    GraspControl.srv
    ManageMoveitScene.srv
    MoveArm.srv
    PalmGoalPoseWorld.srv
    RecordGraspDataSim.srv
    SaveVisualData.srv
    SegmentGraspObject.srv
    UpdateObjectGazebo.srv
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
