#!/usr/bin/env python
import rospy
from std_msgs.msg import String


def publish_point_cloud_paths(scene_point_cloud_path, object_point_cloud_path):
    scene_pub = rospy.Publisher('/scene_point_cloud_path',
                                String,
                                queue_size=10)
    object_pub = rospy.Publisher('/object_point_cloud_path',
                                 String,
                                 queue_size=10)
    rospy.init_node('point_cloud_save_publisher_node')
    rate = rospy.Rate(2)  # 2hz
    while not rospy.is_shutdown():
        scene_pub.publish(scene_point_cloud_path)
        object_pub.publish(object_point_cloud_path)
        rate.sleep()


if __name__ == '__main__':
    scene_point_cloud_path = rospy.get_param('scene_point_cloud_path', None)
    object_point_cloud_path = rospy.get_param('object_point_cloud_path', None)
    try:
        publish_point_cloud_paths(scene_point_cloud_path,
                                  object_point_cloud_path)
    except rospy.ROSInterruptException:
        pass