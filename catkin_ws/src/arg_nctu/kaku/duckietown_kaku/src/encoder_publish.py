#!/usr/bin/env python
import rospy
import roscpp
import numpy as np
import math
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Int64
import tf
import sys
import time
import tf2_ros
class encoder_publisher(object):
	def __init__(self):
		self.node_name = rospy.get_name() 
		self.scale= rospy.get_param("~scale")
		self.shift_x = rospy.get_param("~shift_x")
		self.shift_y = rospy.get_param("~shift_y")
		self.line_marker = Marker()
		self.color =ColorRGBA()
		self.set_line_marker()
		
		self.x = 0
		self.y = 0
		self.th = 0
		self.d_PerCount = (3.14159265 * 0.13) / 5940;
		self.Length =  0.18
		self.previousR = 0
		self.previousL = 0

		self.pub_state_pose_ = tf2_ros.TransformBroadcaster()
		self.pub_marker = rospy.Publisher('~point_visualizer', Marker, queue_size=1)
		# Subscription
		self.sub_encoder = rospy.Subscriber("/encoder", Point, self.cbEncoder, queue_size=1)
		# safe shutdown
		rospy.on_shutdown(self.custom_shutdown)

		# timer
		rospy.loginfo("[%s] Initialized " %(rospy.get_name()))

	def custom_shutdown(self):

		rospy.loginfo("[%s] Shutting down..." %self.node_name)
		rospy.sleep(0.5) #To make sure that it gets published.
		rospy.loginfo("[%s] Shutdown" %self.node_name)

	def cbEncoder(self, encoder_msg):

		delta_R = (encoder_msg.x - previousR) * self.d_PerCount
		delta_L= (encoder_msg.y - previousL) * self.d_PerCount
		delta_S = (delta_R + delta_L)/2

		self.th = (delta_R - delta_L)/self.Length + self.th
		self.x = delta_S * math.cos(self.th) + self.x
		self.y = delta_S * math.sin(self.th) + self.y
		




		
		trans_msg = TransformStamped()
		trans_msg.header.frame_id = 'world'
		trans_msg.child_frame_id = 'duckiebot'
		trans_msg.transform.translation.x = self.x * self.scale + self.shift_x
		trans_msg.transform.translation.y = self.y * self.scale + self.shift_y
		trans_msg.transform.translation.z = 0.1

		new_quaternion = tf.transformations.quaternion_from_euler(0, 0, self.th)
		trans_msg.transform.rotation.x = new_quaternion[0]
		trans_msg.transform.rotation.y = new_quaternion[1]
		trans_msg.transform.rotation.z = new_quaternion[2]
		trans_msg.transform.rotation.w = new_quaternion[3]
		self.pub_state_pose_.sendTransform(trans_msg)

		point = Point()	
		point.x = trans_msg.transform.translation.x 
		point.y = trans_msg.transform.translation.y 
		point.z = trans_msg.transform.translation.z
	

	
		self.line_marker.header.stamp = rospy.Time.now()
		self.line_marker.points.append(point)
		self.line_marker.colors.append(self.color)
		self.pub_marker.publish(self.line_marker)

		previousR = encoder_msg.x
		previousL = encoder_msg.y


	def set_line_marker(self):
		self.line_marker.header.frame_id = "world"
		#marker.ns = namespace
		self.line_marker.action = Marker.ADD
		self.line_marker.id = 0
		self.line_marker.type = 4   
		self.color.r = 0.0
		self.color.g = 1.0
		self.color.b = 0.0
		self.color.a = 1
		if self.tango == "true":
			self.line_marker.scale.x = 0.075
		else:
			self.line_marker.scale.x = 0.01
if __name__ == "__main__":
	rospy.init_node("encoder_publisher",anonymous=False)
	encoder_publisher = encoder_publisher()
	rospy.spin()
