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
    pcd_topic = rospy.get_param('scene_pcd_topic')
    if pcd_topic == '/camera/depth/points':
        pcd_frame = 'camera_depth_optical_frame'
    elif pcd_topic == '/depth_registered_points':
        pcd_frame = 'camera_color_optical_frame'
    else:
        rospy.logerr('Wrong parameter set for scene_pcd_topic in grasp_pipeline_servers.launch')

    transform_pub = rospy.Publisher('/' + pcd_frame + '_in_world', PoseStamped, queue_size=5)
    pose_camera_frame = PoseStamped()
    rospy.sleep(0.5)
    trans = tfBuffer.lookup_transform(pcd_frame, 'world', rospy.Time())
    #pose_camera_frame.header = '/world'
    pose_camera_frame.pose.position = trans.transform.translation
    pose_camera_frame.pose.orientation = trans.transform.rotation
    while not rospy.is_shutdown():
        transform_pub.publish(pose_camera_frame)
        rate.sleep()