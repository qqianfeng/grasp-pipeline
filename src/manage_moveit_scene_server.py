#!/usr/bin/env python
import sys
import rospy
from grasp_pipeline.srv import *
from moveit_commander import RobotCommander, PlanningSceneInterface, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
import roslib.packages as rp


class ManageSceneInMoveit():
    def __init__(self):
        rospy.init_node('manage_moveit_scene_node')
        self.use_sim = rospy.get_param('~sim')

    def handle_create_moveit_scene(self, req):
        scene = PlanningSceneInterface()
        rospy.sleep(0.5)
        print req.object_mesh_path
        scene.add_mesh('obj_mesh', req.object_pose_world, req.object_mesh_path)
        rospy.sleep(4)

        response = ManageMoveitSceneResponse()
        response.success = True
        return response

    def handle_clean_moveit_scene(self, req):
        scene = PlanningSceneInterface()
        rospy.sleep(1)
        scene.remove_world_object('obj_mesh')
        rospy.sleep(2)

        response = ManageMoveitSceneResponse()
        response.success = True
        return response

    def handle_update_moveit_scene(self, req):
        scene = PlanningSceneInterface()
        rospy.sleep(0.1)
        scene.remove_world_object('obj_mesh')
        rospy.sleep(0.1)
        scene.add_mesh('obj_mesh', req.object_pose_world, req.object_mesh_path)
        rospy.sleep(0.1)
        response = ManageMoveitSceneResponse()
        response.success = True
        return response

    def create_create_moveit_scene_server(self):
        rospy.Service('create_moveit_scene', ManageMoveitScene, self.handle_create_moveit_scene)
        rospy.loginfo('Service create_scene:')
        rospy.loginfo('Ready to create the moveit_scene.')

    def create_clean_moveit_scene_server(self):
        rospy.Service('clean_moveit_scene', ManageMoveitScene, self.handle_clean_moveit_scene)
        rospy.loginfo('Service clean_scene:')
        rospy.loginfo('Ready to clean the moveit_scene.')

    def create_update_moveit_scene_server(self):
        rospy.Service('update_moveit_scene', ManageMoveitScene, self.handle_update_moveit_scene)
        rospy.loginfo('Service update_moveit_scene:')
        rospy.loginfo('Ready to update the moveit_scene.')


if __name__ == '__main__':
    ms = ManageSceneInMoveit()
    #ms.create_create_moveit_scene_server()
    #ms.create_clean_moveit_scene_server()
    ms.create_update_moveit_scene_server()
    rospy.spin()
