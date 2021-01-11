#!/usr/bin/env python
import rospy
from std_msgs.msg import String


def publish_pcd_paths(scene_pcd_path, object_pcd_path):
    scene_pub = rospy.Publisher('/scene_pcd_path', String, queue_size=1)
    object_pub = rospy.Publisher('/object_pcd_path', String, queue_size=1)
    rospy.init_node('pcd_save_publisher_node')
    rospy.loginfo('Ready to publish paths, pcd_save_publisher_node started')
    rate = rospy.Rate(2)  # 2hz
    while not rospy.is_shutdown():
        scene_pub.publish(scene_pcd_path)
        object_pub.publish(object_pcd_path)
        rate.sleep()


if __name__ == '__main__':
    scene_pcd_path = rospy.get_param('/pcd_save_publisher_node/scene_pcd_path', None)
    object_pcd_path = rospy.get_param('/pcd_save_publisher_node/object_pcd_path', None)
    rospy.loginfo('Received scene_pcd_path')
    rospy.loginfo(scene_pcd_path)
    if scene_pcd_path is None or object_pcd_path is None:
        rospy.logerr('Point cloud paths could not be read from the parameter server')
    try:
        publish_pcd_paths(scene_pcd_path, object_pcd_path)
    except rospy.ROSInterruptException:
        pass