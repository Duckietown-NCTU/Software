/*
 * Copyright (c) 2010, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// %Tag(FULLTEXT)%
// %Tag(INCLUDES)%
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
//#include <visualization_msgs/MarkerArray.h>
// %EndTag(INCLUDES)%

// %Tag(INIT)%
int main( int argc, char** argv )
{
  ros::init(argc, argv, "basic_shapes");
  ros::NodeHandle n;
  ros::Rate r(1);
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);
// %EndTag(INIT)%

  // Set our initial shape type to be a cube
// %Tag(SHAPE_INIT)%
  uint32_t arrow = visualization_msgs::Marker::ARROW;
  std::string FRAME_ID = "/my_frame";
// %EndTag(SHAPE_INIT)%

// %Tag(MARKER_INIT)%
  while (ros::ok())
  {
    visualization_msgs::Marker marker;
    visualization_msgs::Marker marker2;
    //visualization_msgs::Marker marker3;
    //visualization_msgs::Marker marker4;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "/my_frame";
    marker.header.stamp = ros::Time::now();

    marker2.header.frame_id = FRAME_ID;
    marker2.header.stamp = ros::Time::now();

    //    marker3.header.frame_id = FRAME_ID;
    //marker3.header.stamp = ros::Time::now();

    //    marker4.header.frame_id = FRAME_ID;
    //marker4.header.stamp = ros::Time::now();
// %EndTag(MARKER_INIT)%

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
// %Tag(NS_ID)%
    marker.ns = "basic_shapes";
    marker.id = 0;

    marker2.ns = "basic_shapes";
    marker2.id = 1;

    //marker3.ns = "basic_shapes";
    //marker3.id = 2;

    //marker4.ns = "basic_shapes";
    //marker4.id = 3;
// %EndTag(NS_ID)%

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
// %Tag(TYPE)%
    marker.type = arrow;
    marker2.type = arrow;
    //marker3.type = arrow;
    //marker4.type = arrow;
// %EndTag(TYPE)%

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
// %Tag(ACTION)%
    marker.action = visualization_msgs::Marker::ADD;
    marker2.action = visualization_msgs::Marker::ADD;
    //marker3.action = visualization_msgs::Marker::ADD;
    //marker4.action = visualization_msgs::Marker::ADD;
// %EndTag(ACTION)%

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
// %Tag(POSE)%
    marker.pose.position.x = -0.75;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0;
    marker.pose.orientation.w = 1.0;
// %EndTag(POSE)%

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
// %Tag(SCALE)%
    marker.scale.x = 1.0;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
// %EndTag(SCALE)%

    // Set the color -- be sure to set alpha to something non-zero!
// %Tag(COLOR)%
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;
    
    marker2.pose.position.x = 0;
    marker2.pose.position.y = 0.75;
    marker2.pose.position.z = 0.0;
    marker2.pose.orientation.x = 0.0;
    marker2.pose.orientation.y = 0.0;
    marker2.pose.orientation.z = 0.0;
    marker2.pose.orientation.w = 0.0;
// %EndTag(POSE)%

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
// %Tag(SCALE)%
    marker2.scale.x = 1.0;
    marker2.scale.y = 0.2;
    marker2.scale.z = 0.2;
// %EndTag(SCALE)%

    // Set the color -- be sure to set alpha to something non-zero!
// %Tag(COLOR)%
    marker2.color.r = 0.0f;
    marker2.color.g = 1.0f;
    marker2.color.b = 0.0f;
    marker2.color.a = 1.0;

    /*
    marker3.pose.position.x = 0.75;
    marker3.pose.position.y = 0;
    marker3.pose.position.z = 1.0;
    marker3.pose.orientation.x = -1.0;
    marker3.pose.orientation.y = -1.0;
    marker3.pose.orientation.z = 0.0;
    marker3.pose.orientation.w = 1.0;
// %EndTag(POSE)%

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
// %Tag(SCALE)%
    marker3.scale.x = 1.0;
    marker3.scale.y = 1.0;
    marker3.scale.z = 0.3;
// %EndTag(SCALE)%

    // Set the color -- be sure to set alpha to something non-zero!
// %Tag(COLOR)%
    marker3.color.r = 0.0f;
    marker3.color.g = 0.0f;
    marker3.color.b = 1.0f;
    marker3.color.a = 1.0;
*/
    //marker4.pose.position.x = 0.0;
    //marker4.pose.position.y = -0.75;
    //marker4.pose.position.z = 0.0;
    //marker4.pose.orientation.x = -1.0;
    //marker4.pose.orientation.y = 1.0;
    //marker4.pose.orientation.z = 0.0;
    //marker4.pose.orientation.w = 1.0;
// %EndTag(POSE)%

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
// %Tag(SCALE)%
    //marker4.scale.x = 1.0;
    //marker4.scale.y = 1.0;
    //marker4.scale.z = 0.4;
// %EndTag(SCALE)%

    // Set the color -- be sure to set alpha to something non-zero!
// %Tag(COLOR)%
    // marker4.color.r = 1.0f;
    //marker4.color.g = 1.0f;
    //marker4.color.b = 0.0f;
    //marker4.color.a = 1.0;
// %EndTag(COLOR)%

// %Tag(LIFETIME)%
    marker.lifetime = ros::Duration(); 
    marker2.lifetime = ros::Duration();
    //marker3.lifetime = ros::Duration(); 
    //marker4.lifetime = ros::Duration();
// %EndTag(LIFETIME)%

    // Publish the marker
// %Tag(PUBLISH)%
    while (marker_pub.getNumSubscribers() < 1)
    {
      if (!ros::ok())
      {
        return 0;
      }
      ROS_WARN_ONCE("Please create a subscriber to the marker");
      sleep(1);
    }
    //visualization_msgs::MarkerArray tile;
    //tile.markers[0] = marker;
    //tile.markers[1] = marker2;
    //tile.markers[2] = marker3;
    //tile.markers[3] = marker4;
    marker_pub.publish(marker);
    marker_pub.publish(marker2);
    //marker_pub.publish(marker3);
    //marker_pub.publish(marker4);
// %EndTag(PUBLISH)%

// %Tag(SLEEP_END)%
    r.sleep();
  }
// %EndTag(SLEEP_END)%
}
// %EndTag(FULLTEXT)%
