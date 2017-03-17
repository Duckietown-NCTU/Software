#!/usr/bin/env python
import rospy
import roscpp
import numpy as np
import math
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf
import sys
import time
import tf2_ros
class state_pose_publisher(object):
	def __init__(self):
		self.node_name = rospy.get_name() 
		self.tango = rospy.get_param("~tango")
		self.scale= rospy.get_param("~scale")
		self.shift_x = rospy.get_param("~shift_x")
		self.shift_y = rospy.get_param("~shift_y")
		self.line_marker = Marker()
		self.color =ColorRGBA()
		self.set_line_marker()
		#print 1
		# Publicaiton
		#self.pub_state_pose_ = rospy.Publisher("~car_cmd",Twist2DStamped,queue_size=1)
		self.pub_state_pose_ = tf2_ros.TransformBroadcaster()
		self.pub_marker = rospy.Publisher('~point_visualizer', Marker, queue_size=1)
		# Subscription
		self.sub_pose_ = rospy.Subscriber("~pose", Pose, self.cbPose, queue_size=1)
		# safe shutdown
		rospy.on_shutdown(self.custom_shutdown)

		# timer
		rospy.loginfo("[%s] Initialized " %(rospy.get_name()))

	def custom_shutdown(self):

		rospy.loginfo("[%s] Shutting down..." %self.node_name)
		rospy.sleep(0.5) #To make sure that it gets published.
		rospy.loginfo("[%s] Shutdown" %self.node_name)

	def cbPose(self, pose_msg):
		#print 1
		trans_msg = TransformStamped()
		trans_msg.header.frame_id = 'world'
		trans_msg.child_frame_id = 'duckiebot'
		trans_msg.transform.translation.x = pose_msg.position.x * self.scale + self.shift_x
		trans_msg.transform.translation.y = pose_msg.position.y * self.scale + self.shift_y
		#trans_msg.transform.translation.z = pose_msg.position.z
		#trans_msg.transform.translation.x = pose_msg.position.x * self.scale
		#trans_msg.transform.translation.y = pose_msg.position.y * self.scale
		trans_msg.transform.translation.z = 0.1
		print 'x: ', trans_msg.transform.translation.x, ' y: ', trans_msg.transform.translation.y

		quaternion = ( pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)
		#print euler

		if self.tango == "false":
			roll = math.pi - euler[2] 
			pitch = math.pi - euler[1] 
			yaw =  math.pi*2 - euler[0]
			#print roll, pitch, yaw
		else:
			#print "tango"
			roll = math.pi - euler[1] 
			#print "r : " , euler[1] , " p : " , euler[2] , "y : " , math.pi*2 - euler[0]
			pitch = math.pi - euler[2] 
			yaw =  math.pi*2 - euler[0]

		new_quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
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
	rospy.init_node("state_pose_publisher",anonymous=False)
	state_pose_publisher_node = state_pose_publisher()
	rospy.spin()
