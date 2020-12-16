#!/usr/bin/env python
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
import numpy as np

if __name__ == '__main__':
    rospy.init_node('camera_transform_broadcaster_node')
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rate = rospy.Rate(1)
    transform_pub = rospy.Publisher('/camera_color_optical_frame_in_world',
                                    PoseStamped,
                                    queue_size=5)
    pose_color = PoseStamped()
    rospy.sleep(0.5)
    trans = tfBuffer.lookup_transform('camera_color_optical_frame', 'world',
                                      rospy.Time())
    #pose_color.header = '/world'
    pose_color.pose.position = trans.transform.translation
    pose_color.pose.orientation = trans.transform.rotation
    while not rospy.is_shutdown():
        transform_pub.publish(pose_color)
        rate.sleep()

# WIE MACHEN DIE UTAH GUYS DAS? Transformieren sie die Pointcloud?; p_w = w_T_c * p_c; DAS GILT