#ifndef APRILTAG_DETECTOR_H
#define APRILTAG_DETECTOR_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <duckietown_msgs/BoolStamped.h>
#include <AprilTags/TagDetector.h>
#include <AprilTags/TagRectDetector.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/CompressedImage.h>
namespace apriltags_ros{


class AprilTagDescription{
 public:
  AprilTagDescription(int id, double size, std::string &frame_name):id_(id), size_(size), frame_name_(frame_name){}
  double size(){return size_;}
  int id(){return id_;} 
  std::string& frame_name(){return frame_name_;} 
 private:
  int id_;
  double size_;
  std::string frame_name_;

};


class AprilTagDetector{
 public:
  AprilTagDetector(ros::NodeHandle& nh, ros::NodeHandle& pnh);
  ~AprilTagDetector();
 private:
void switchCB(const duckietown_msgs::BoolStamped::ConstPtr& switch_msg);
  void imageCb(const sensor_msgs::ImageConstPtr& msg,const sensor_msgs::CameraInfoConstPtr& cam_info);
  void image_compress_Cb(const sensor_msgs::CompressedImageConstPtr& msg);
  std::map<int, AprilTagDescription> parse_tag_descriptions(XmlRpc::XmlRpcValue& april_tag_descriptions);

 private:
  std::map<int, AprilTagDescription> descriptions_;
  std::string sensor_frame_id_;
  image_transport::ImageTransport it_;
  //image_transport::ImageTransport it2_;
  image_transport::CameraSubscriber image_sub_;
  image_transport::Publisher image_pub_;
  image_transport::Publisher crop_image_pub_;
  ros::Subscriber switch_sub_;
  ros::Publisher detections_pub_;
  ros::Publisher proposals_pub_;
  ros::Publisher pose_pub_;
  ros::Subscriber image_compress_sub_;
  tf::TransformBroadcaster tf_pub_;
  boost::shared_ptr<AprilTags::TagDetector> tag_detector_;
  boost::shared_ptr<AprilTags::TagRectDetector> tag_rect_detector_;
  bool on_switch;
};



}


#endif
