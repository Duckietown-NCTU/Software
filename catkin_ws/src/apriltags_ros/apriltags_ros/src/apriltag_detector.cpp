#include <apriltags_ros/apriltag_detector.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/foreach.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <duckietown_msgs/AprilTagDetection.h>
#include <duckietown_msgs/AprilTagDetectionArray.h>
#include <duckietown_msgs/Rect.h>
#include <duckietown_msgs/Rects.h>
#include <AprilTags/Tag16h5.h>
#include <AprilTags/Tag25h7.h>
#include <AprilTags/Tag25h9.h>
#include <AprilTags/Tag36h9.h>
#include <AprilTags/Tag36h11.h>
#include <XmlRpcException.h>
#include <time.h>

namespace apriltags_ros{

  AprilTagDetector::AprilTagDetector(ros::NodeHandle& nh, ros::NodeHandle& pnh): it_(nh){
    XmlRpc::XmlRpcValue april_tag_descriptions;
    if(!pnh.getParam("tag_descriptions", april_tag_descriptions)){
      ROS_WARN("No april tags specified");
    }
    else{
      try{
	descriptions_ = parse_tag_descriptions(april_tag_descriptions);
      } catch(XmlRpc::XmlRpcException e){
	ROS_ERROR_STREAM("Error loading tag descriptions: "<<e.getMessage());
      }
    }
    
    if(!pnh.getParam("sensor_frame_id", sensor_frame_id_)){
      sensor_frame_id_ = "";
    }
    
    AprilTags::TagCodes tag_codes = AprilTags::tagCodes36h11;
    tag_detector_= boost::shared_ptr<AprilTags::TagDetector>(new AprilTags::TagDetector(tag_codes));
    tag_rect_detector_= boost::shared_ptr<AprilTags::TagRectDetector>(new AprilTags::TagRectDetector(tag_codes));
    image_sub_ = it_.subscribeCamera("image_rect", 1, &AprilTagDetector::imageCb, this);
    switch_sub_ = nh.subscribe("switch",1,&AprilTagDetector::switchCB, this);
    image_pub_ = it_.advertise("tag_detections_image", 1);
    detections_pub_ = nh.advertise<duckietown_msgs::AprilTagDetectionArray>("tag_detections", 1);
    proposals_pub_ = nh.advertise<duckietown_msgs::Rects>("quad_proposals", 1);
    image_compress_sub_ = nh.subscribe("/wama/camera_node/image/compressed", 1, &AprilTagDetector::image_compress_Cb, this);
    //crop_image_pub_ = it2_.advertise("crop_image", 1);
    pose_pub_ = nh.advertise<geometry_msgs::PoseArray>("tag_detections_pose", 1);
    on_switch=false;
  }
  AprilTagDetector::~AprilTagDetector(){
    image_sub_.shutdown();
  }

  void AprilTagDetector::switchCB(const duckietown_msgs::BoolStamped::ConstPtr& switch_msg){
    on_switch=switch_msg->data;
  }
  
  void AprilTagDetector::imageCb(const sensor_msgs::ImageConstPtr& msg,const sensor_msgs::CameraInfoConstPtr& cam_info){
    cv_bridge::CvImagePtr cv_ptr, cv_crop;
    try{
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat gray;
    cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);
    //std::vector<AprilTags::TagDetection>	detections = tag_detector_->extractTags(gray);
    //std::vector<AprilTags::TagDetection>	detections = tag_rect_detector_->extractTags(gray);
    std::vector<cv::Rect> quad_proposals;
    std::vector<AprilTags::TagDetection>	detections = tag_rect_detector_->extractTags(gray, quad_proposals);
    
    ROS_DEBUG("%d tag detected", (int)detections.size());
    ROS_DEBUG("%d quad detected", (int)quad_proposals.size());
    
    double fx = cam_info->K[0];
    double fy = cam_info->K[4];
    double px = cam_info->K[2];
    double py = cam_info->K[5];
    
    if(!sensor_frame_id_.empty())
      cv_ptr->header.frame_id = sensor_frame_id_;
      //cv_crop->header.frame_id = sensor_frame_id_;
    
    duckietown_msgs::AprilTagDetectionArray tag_detection_array;
    duckietown_msgs::Rects quad_proposal_array;
    geometry_msgs::PoseArray tag_pose_array;
    tag_pose_array.header = cv_ptr->header;
    
    BOOST_FOREACH(AprilTags::TagDetection detection, detections){
      std::map<int, AprilTagDescription>::const_iterator description_itr = descriptions_.find(detection.id);
      if(description_itr == descriptions_.end()){
	ROS_WARN_THROTTLE(10.0, "Found tag: %d, but no description was found for it", detection.id);
	continue;
      }
      AprilTagDescription description = description_itr->second;
      double tag_size = description.size();
      
      detection.draw(cv_ptr->image);
      Eigen::Matrix4d transform = detection.getRelativeTransform(tag_size, fx, fy, px, py);
      Eigen::Matrix3d rot = transform.block(0,0,3,3);
      Eigen::Quaternion<double> rot_quaternion = Eigen::Quaternion<double>(rot);

      geometry_msgs::PoseStamped tag_pose;
      tag_pose.pose.position.x = transform(0,3);
      tag_pose.pose.position.y = transform(1,3);
      tag_pose.pose.position.z = transform(2,3);
      tag_pose.pose.orientation.x = rot_quaternion.x();
      tag_pose.pose.orientation.y = rot_quaternion.y();
      tag_pose.pose.orientation.z = rot_quaternion.z();
      tag_pose.pose.orientation.w = rot_quaternion.w();
      tag_pose.header = cv_ptr->header;

      duckietown_msgs::AprilTagDetection tag_detection;
      tag_detection.pose = tag_pose;
      tag_detection.id = detection.id;
      tag_detection.size = tag_size;
      tag_detection_array.detections.push_back(tag_detection);
      tag_pose_array.poses.push_back(tag_pose.pose);

      tf::Stamped<tf::Transform> tag_transform;
      tf::poseStampedMsgToTF(tag_pose, tag_transform);
      tf_pub_.sendTransform(tf::StampedTransform(tag_transform, tag_transform.stamp_, tag_transform.frame_id_, description.frame_name()));
    }

	cv::Rect crop_proposal;
	cv_crop = cv_ptr;
    for(int qi = 0; qi < quad_proposals.size(); qi++){
      duckietown_msgs::Rect quad_proposal;
      quad_proposal.x = quad_proposals[qi].x;
      quad_proposal.y = quad_proposals[qi].y;
      quad_proposal.w = quad_proposals[qi].width;
      quad_proposal.h = quad_proposals[qi].height;
      
	  // publish crop image
	  //crop_proposal.x = quad_proposals[qi].x;
	  //crop_proposal.y = quad_proposals[qi].y;
	  //crop_proposal.width = quad_proposals[qi].width;
	  //crop_proposal.height = quad_proposals[qi].height;
	  crop_proposal.x = 5;
	  crop_proposal.y = 5;
	  crop_proposal.width = 20;
	  crop_proposal.height = 20;
	  //cv_crop->image = cv_ptr->image.clone();
	  //cv_crop->image = cv_ptr->image(crop_proposal).clone();
	  //crop_image_pub_.publish(cv_crop->toImageMsg());
	
	  //image_pub_.publish(cv_crop->toImageMsg());
	  //visualization
      //cv::rectangle(cv_ptr->image, quad_proposals[qi], cv::Scalar(255,0,0));
      quad_proposal_array.rects.push_back(quad_proposal);
    }

    detections_pub_.publish(tag_detection_array);
    proposals_pub_.publish(quad_proposal_array);
    pose_pub_.publish(tag_pose_array);
	image_pub_.publish(cv_crop->toImageMsg());
    //image_pub_.publish(cv_ptr->toImageMsg());
  }

  void AprilTagDetector::image_compress_Cb(const sensor_msgs::CompressedImageConstPtr& msg){
    // measure process time
    cout << "start" << endl;
    const clock_t begin_time = clock();
    //
    if(on_switch == false){
      return;
    }
    cv_bridge::CvImagePtr cv_ptr, cv_crop;
    try{
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat gray;
    cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);
    //std::vector<AprilTags::TagDetection>  detections = tag_detector_->extractTags(gray);
    //std::vector<AprilTags::TagDetection>  detections = tag_rect_detector_->extractTags(gray);
    std::vector<cv::Rect> quad_proposals;
    std::vector<AprilTags::TagDetection>  detections = tag_rect_detector_->extractTags(gray, quad_proposals);
    
    ROS_DEBUG("%d tag detected", (int)detections.size());
    ROS_DEBUG("%d quad detected", (int)quad_proposals.size());
    
    double fx = 319.7232254665968;
    double fy = 296.50332879412235;
    double px = 324.92020966876265;
    double py = 231.20049166797017;
    
    if(!sensor_frame_id_.empty())
      cv_ptr->header.frame_id = sensor_frame_id_;
      //cv_crop->header.frame_id = sensor_frame_id_;
    
    duckietown_msgs::AprilTagDetectionArray tag_detection_array;
    duckietown_msgs::Rects quad_proposal_array;
    geometry_msgs::PoseArray tag_pose_array;
    tag_pose_array.header = cv_ptr->header;
    
    BOOST_FOREACH(AprilTags::TagDetection detection, detections){
      std::map<int, AprilTagDescription>::const_iterator description_itr = descriptions_.find(detection.id);
      if(description_itr == descriptions_.end()){
  ROS_WARN_THROTTLE(10.0, "Found tag: %d, but no description was found for it", detection.id);
  continue;
      }
      AprilTagDescription description = description_itr->second;
      double tag_size = description.size();
      
      detection.draw(cv_ptr->image);
      Eigen::Matrix4d transform = detection.getRelativeTransform(tag_size, fx, fy, px, py);
      Eigen::Matrix3d rot = transform.block(0,0,3,3);
      Eigen::Quaternion<double> rot_quaternion = Eigen::Quaternion<double>(rot);

      geometry_msgs::PoseStamped tag_pose;
      tag_pose.pose.position.x = transform(0,3);
      tag_pose.pose.position.y = transform(1,3);
      tag_pose.pose.position.z = transform(2,3);
      tag_pose.pose.orientation.x = rot_quaternion.x();
      tag_pose.pose.orientation.y = rot_quaternion.y();
      tag_pose.pose.orientation.z = rot_quaternion.z();
      tag_pose.pose.orientation.w = rot_quaternion.w();
      tag_pose.header = cv_ptr->header;

      duckietown_msgs::AprilTagDetection tag_detection;
      tag_detection.pose = tag_pose;
      tag_detection.id = detection.id;
      tag_detection.size = tag_size;
      tag_detection_array.detections.push_back(tag_detection);
      tag_pose_array.poses.push_back(tag_pose.pose);

      tf::Stamped<tf::Transform> tag_transform;
      tf::poseStampedMsgToTF(tag_pose, tag_transform);
      tf_pub_.sendTransform(tf::StampedTransform(tag_transform, tag_transform.stamp_, tag_transform.frame_id_, description.frame_name()));
    }

    cv::Rect crop_proposal;
    cv_crop = cv_ptr;
    for(int qi = 0; qi < quad_proposals.size(); qi++){
      duckietown_msgs::Rect quad_proposal;
      quad_proposal.x = quad_proposals[qi].x;
      quad_proposal.y = quad_proposals[qi].y;
      quad_proposal.w = quad_proposals[qi].width;
      quad_proposal.h = quad_proposals[qi].height;
      
    // publish crop image
    //crop_proposal.x = quad_proposals[qi].x;
    //crop_proposal.y = quad_proposals[qi].y;
    //crop_proposal.width = quad_proposals[qi].width;
    //crop_proposal.height = quad_proposals[qi].height;
    crop_proposal.x = 5;
    crop_proposal.y = 5;
    crop_proposal.width = 20;
    crop_proposal.height = 20;
    //cv_crop->image = cv_ptr->image.clone();
    //cv_crop->image = cv_ptr->image(crop_proposal).clone();
    //crop_image_pub_.publish(cv_crop->toImageMsg());
  
    //image_pub_.publish(cv_crop->toImageMsg());
    //visualization
      //cv::rectangle(cv_ptr->image, quad_proposals[qi], cv::Scalar(255,0,0));
      quad_proposal_array.rects.push_back(quad_proposal);
    }

    detections_pub_.publish(tag_detection_array);
    proposals_pub_.publish(quad_proposal_array);
    pose_pub_.publish(tag_pose_array);
    image_pub_.publish(cv_crop->toImageMsg());
    //image_pub_.publish(cv_ptr->toImageMsg());
    // measure process time
    cout << "end" << endl;
    cout << "processtime : "<< float( clock () - begin_time ) << endl;
    //
  }


  std::map<int, AprilTagDescription> AprilTagDetector::parse_tag_descriptions(XmlRpc::XmlRpcValue& tag_descriptions){
    std::map<int, AprilTagDescription> descriptions;
    ROS_ASSERT(tag_descriptions.getType() == XmlRpc::XmlRpcValue::TypeArray);
    for (int32_t i = 0; i < tag_descriptions.size(); ++i) {
      XmlRpc::XmlRpcValue& tag_description = tag_descriptions[i];
      ROS_ASSERT(tag_description.getType() == XmlRpc::XmlRpcValue::TypeStruct);
      ROS_ASSERT(tag_description["id"].getType() == XmlRpc::XmlRpcValue::TypeInt);
      ROS_ASSERT(tag_description["size"].getType() == XmlRpc::XmlRpcValue::TypeDouble);

      int id = (int)tag_description["id"];
      double size = (double)tag_description["size"];

      std::string frame_name;
      if(tag_description.hasMember("frame_id")){
	ROS_ASSERT(tag_description["frame_id"].getType() == XmlRpc::XmlRpcValue::TypeString);
	frame_name = (std::string)tag_description["frame_id"];
      }
      else{
	std::stringstream frame_name_stream;
	frame_name_stream << "tag_" << id;
	frame_name = frame_name_stream.str();
      }
      AprilTagDescription description(id, size, frame_name);
      ROS_INFO_STREAM("Loaded tag config: "<<id<<", size: "<<size<<", frame_name: "<<frame_name);
      descriptions.insert(std::make_pair(id, description));
    }
    return descriptions;
  }


}
