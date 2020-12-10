#!/usr/bin/env python
import rospy
from std_msgs.msg import String


def publish_point_cloud_paths(scene_point_cloud_path, object_point_cloud_path):
    scene_pub = rospy.Publisher('/scene_point_cloud_path',
                                String,
                                queue_size=1)
    object_pub = rospy.Publisher('/object_point_cloud_path',
                                 String,
                                 queue_size=1)
    rospy.init_node('point_cloud_save_publisher_node')
    rospy.loginfo(
        'Ready to publish paths, point_cloud_save_publisher_node started')
    rate = rospy.Rate(2)  # 2hz
    while not rospy.is_shutdown():
        scene_pub.publish(scene_point_cloud_path)
        object_pub.publish(object_point_cloud_path)
        rate.sleep()


if __name__ == '__main__':
    scene_point_cloud_path = rospy.get_param(
        '/point_cloud_save_publisher_node/scene_point_cloud_path', None)
    object_point_cloud_path = rospy.get_param(
        '/point_cloud_save_publisher_node/object_point_cloud_path', None)
    rospy.loginfo('Received scene_point_cloud_path')
    rospy.loginfo(scene_point_cloud_path)
    if scene_point_cloud_path is None or object_point_cloud_path is None:
        rospy.logerr(
            'Point cloud paths could not be read from the parameter server')
    try:
        publish_point_cloud_paths(scene_point_cloud_path,
                                  object_point_cloud_path)
    except rospy.ROSInterruptException:
        pass