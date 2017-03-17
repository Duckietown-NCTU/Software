#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
using namespace cv;
using namespace std;


image_transport::Publisher image_pub;
	
void chatterCallback(sensor_msgs::CompressedImage image_msg)
//void chatterCallback(sensor_msgs::Compr msg)
{
//
//	cv_bridge::CvImagePtr cv_ptr;
//	cv_ptr = cv_bridge::toCvCopy(image_msg,sensor_msgs::image_encodings::BGR8);
//
//	
//	if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
//		circle(cv_ptr->image, Point(150,150), 40, CV_RGB(255,0,0));	
//
//	namedWindow("Camera");
//  	imshow("Camera",cv_ptr->image);
//  	waitKey(1);
//	
//	image_pub.publish(cv_ptr->toImageMsg());
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "imgpreproc");
  ros::NodeHandle n;
  image_transport::ImageTransport it(n);
  
  image_pub = it.advertise("/image_converter/output_video",1);
   
  ros::Subscriber sub = n.subscribe("/grasseater/camera_node/image/compressed",1, chatterCallback);

  /**
   * ros::spin() will enter a loop, pumping callbacks.  With this version, all
   * callbacks will be called from within this thread (the main one).  ros::spin()
   * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
   */
  ros::spin();

  return 0;
}
