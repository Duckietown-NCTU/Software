#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

//This node subscribes to /grasseater/camera_node/image/compressed and undistort the image. Then it publish the image to imgpreproc/undist

using namespace cv;
using namespace std;

const double fx=259.5050;
const double fy=263.0909;
const double cx=333.0998;
const double cy=254.7687;
const double s=0.3627;
const double dist[]={-0.2912,0.0630,-0.0051,-0.00062276,0.0017};

//Calibration Parameters
    Mat cameraMatrix = (Mat_<double>(3,3) << fx,s,cx,0.0,fy,cy,0.0,0.0,1.0);
    Mat distParamCV = (Mat_<double>(1,5) << dist[0],dist[1],dist[2],dist[3],dist[4]);
//Create a Mat object to store the undistorted image
    Mat undistImg;

image_transport::Publisher image_pub;
	
void chatterCallback(sensor_msgs::CompressedImage image_msg)
{
    ros::Time time = ros::Time::now();

//    cv_bridge::CvImagePtr cv_ptr;
//    cv_ptr = cv_bridge::toCvCopy(image_msg,sensor_msgs::image_encodings::BGR8);
//
//    //Undistort the image
//    undistort(cv_ptr->image,undistImg,cameraMatrix,distParamCV);
//    //Create a cvImage to transform from OpenCVImage to ImageMsg
//    cv_bridge::CvImage cvi_undist;
//    cvi_undist.header.stamp = time;
//    cvi_undist.header.frame_id = "undist";
//    cvi_undist.encoding = "bgr8";
//    cvi_undist.image = undistImg;
//
//    //Publish the image
//    image_pub.publish(cvi_undist.toImageMsg());
//
//    ROS_INFO_ONCE("First image published");
}

//Transform from world coodinates to image coordinates
void w2imCoord(cv::Point2f worldPoint, cv::Point2f &imagePoint, Eigen::MatrixXd homog)
{
    Eigen::Vector3d wPointEig(worldPoint.x,worldPoint.y,1.0);    
    Eigen::Vector3d imPointEig;
    imPointEig=homog*wPointEig;
    imPointEig=imPointEig/imPointEig(2);
    imagePoint=Point2f(imPointEig(0),imPointEig(1));
}

//Transform from image coordinates to world coordinates
void im2wCoord(cv::Point2f imagePoint, cv::Point2f &worldPoint, Eigen::MatrixXd homogInv)
{
    Eigen::Vector3d imPointEig(imagePoint.x,imagePoint.y,1.0);    
    Eigen::Vector3d wPointEig;
    wPointEig=homogInv*imPointEig;
    wPointEig=wPointEig/wPointEig(2);
    worldPoint=Point2f(wPointEig(0),wPointEig(1));
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "imgpreproc");
  ros::NodeHandle n;
  image_transport::ImageTransport it(n);
  
  image_pub = it.advertise("/imgpreproc/undist",1);
   
  ros::Subscriber sub = n.subscribe("/grasseaterpi3/camera_node/image/compressed",1, chatterCallback);

  
  ros::spin();

  return 0;
}
