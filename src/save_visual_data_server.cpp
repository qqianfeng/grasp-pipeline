#include "ros/ros.h"
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <grasp_pipeline/SaveVisualData.h>

#include <iostream>

#include <pcl/common/common_header.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


inline void saveDepthImage(cv::Mat& depth_image, std::string depth_image_save_path){
    cv::Mat depth_save_image;
    depth_save_image.create(depth_image.size(), CV_32FC1);
    depth_image.convertTo(depth_save_image, CV_32FC1);
    imwrite(depth_image_save_path, depth_save_image);
}

bool saveVisualData(grasp_pipeline::SaveVisualData::Request &req, grasp_pipeline::SaveVisualData::Response &res){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(req.scene_cloud, *raw_cloud_ptr);
    pcl::io:savePCDFile(req.scene_cloud_save_path, *raw_cloud_ptr, true);

    cv_bridge::CvImagePtr cv_rgb_msg = cv_bridge::toCvCopy(req.scene_color_img, sensor_msgs::image_encodings::BGR8);
    imwrite(req.colo_image_save_path, cv_rgb_msg->image);

    cv::Mat depth_image;
    cv_bridge::CvImagePtr cv_depth_msg = cv_bridge::toCvCopy(req.scene_depth_img);
    cv_depth_msg->image.convertTo(depth_image, CV_32FC1);
    saveDepthImage(depth_image, req.depth_image_save_path);

    res.save_visual_data_success = true;
    return true;
}

void testDepthImage(const sensor_msgs::ImageConstPtr& depth_msg){
    cv_bridge::CvImagePtr cv_depth_msg = cv_bridge::toCvCopy(depth_msg);
    cv::Mat depth_frame;
    cv_depth_msg->image.convertTo(depth_frame, CV_32FC1);
}


int main(int argc, char **argv){
    ros::init(argc, argv, "save_visual_data_server");
    ros::NodeHandle n;

    ros::ServiceServer service = n.advertiseService("/save_visual_data", saveVisualData);
    ROS_INFO("Service save_visual_data_server:");
    ROS_INFO("Ready to save visual data.");

    //subscribe to image topics and test display, write, save functionality
    image_transport::ImageTransport it(n);
    image_transport::Subscriber sub = it.subscribe("/camera/depth/image_raw", 1, testDepthImage);

    ros::spin();

    return 0;
}
