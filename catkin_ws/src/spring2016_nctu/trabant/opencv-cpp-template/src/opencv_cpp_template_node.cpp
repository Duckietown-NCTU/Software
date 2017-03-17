#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void callback(const sensor_msgs::ImageConstPtr& imageMap)
{
    ROS_INFO("Received image %dx%d", imageMap->width, imageMap->height);
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(imageMap, imageMap->encoding); 

    cv::Mat im_pi = cv::Mat::zeros(480, 640, CV_8UC3);
    im_pi = cv_ptr->image.clone();

}

void callback_compressed(const sensor_msgs::CompressedImageConstPtr& imageMap)
{
    ROS_INFO("Received Compressed Image ");

    cv::Mat im_pi = cv::Mat::zeros(480, 640, CV_8UC3);
    im_pi = cv::imdecode(cv::Mat(imageMap->data), CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(im_pi, im_pi, CV_RGB2BGR);

    //YOU CODE HERE
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "opencv_cpp_template_node");
    ros::NodeHandle nh("~");

    // read param from launch file
    std::string veh;

    // There are two ways to setup the param
    //nh.param<std::string>("veh", veh, "trabant"); // make sure you init nh by "~"
    ros::param::param<std::string>("~veh", veh, "trabant");
    std::stringstream ss;
    ss << "/" << veh << "/camera_node/image/compressed";
    ros::Subscriber sub = nh.subscribe(ss.str().c_str(), 1000, callback_compressed);
    ROS_INFO("Start opencv_cpp_template_node, and subscribe topic: %s", ss.str().c_str());

    ros::spin();
    return 0;
}
