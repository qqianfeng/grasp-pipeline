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
        self.add_ground_plane()

    def add_ground_plane(self):
        rospy.loginfo("Adding ground plane to planning scene.")
        scene = PlanningSceneInterface()
        rospy.sleep(0.5)
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.orientation.w = 1
        scene.add_plane("ground_plane", pose)

        rospy.sleep(0.5)
        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = scene.get_attached_objects(["ground_plane"])
            is_attached = len(attached_objects.keys()) > 0

            is_known = "ground_plane" in scene.get_known_object_names()

            # Test if we are in the expected state
            if is_attached or is_known:
                rospy.loginfo("Ground plane is now known to moveit.")
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.2)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        rospy.loginfo("Ground plane is not known to moveit.")
        return False

    def handle_create_moveit_scene(self, req):
        scene = PlanningSceneInterface()
        rospy.sleep(0.5)
        for idx in range(len(req.object_mesh_paths)):
            scene.add_mesh(req.object_names[idx], req.object_pose_worlds[idx], req.object_mesh_paths[idx])
        rospy.sleep(0.5)

        response = ManageMoveitSceneResponse()
        response.success = True
        return response

    def handle_clean_moveit_scene(self, req):
        scene = PlanningSceneInterface()
        rospy.sleep(0.5)
        for idx in range(len(req.object_mesh_paths)):
            scene.remove_world_object(req.object_names[idx])
        rospy.sleep(0.5)

        response = ManageMoveitSceneResponse()
        response.success = True
        return response

    def handle_update_moveit_scene(self, req):
        scene = PlanningSceneInterface()
        rospy.sleep(0.5)
        for idx in range(len(req.object_mesh_paths)):
            scene.remove_world_object(req.object_names[idx])
        rospy.sleep(0.5)
        for idx in range(len(req.object_mesh_paths)):
            scene.add_mesh(req.object_names[idx], req.object_pose_worlds[idx], req.object_mesh_paths[idx])
        rospy.sleep(0.5)
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
    ms.create_create_moveit_scene_server()
    ms.create_clean_moveit_scene_server()
    ms.create_update_moveit_scene_server()
    rospy.spin()
