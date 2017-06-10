#!/usr/bin/env python
import rospy
import roscpp
import numpy as np
import math
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from gazebo_msgs.msg import ModelState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Int64
import tf
import sys
import time
import tf2_ros
class gazebo_publisher(object):
	def __init__(self):
		self.node_name = rospy.get_name() 
		
		self.x = 0
		self.y = 0
		self.th = 0
		self.d_PerCount = ((3.14159265 * 0.13) / 5940)*0.3/0.635
		self.Length =  0.18
		self.previousR = 0
		self.previousL = 0

		self.pub_gazebo = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
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

		delta_R = (encoder_msg.x - self.previousR) * self.d_PerCount
		delta_L= (encoder_msg.y - self.previousL) * self.d_PerCount
		delta_S = (delta_R + delta_L)/2
		# print "encoder_msg.x = ",encoder_msg.x," encoder_msg.y = ",encoder_msg.y
		#print "previousR = ",self.previousR," previousL = ",self.previousL

		self.th = ((delta_L - delta_R)/self.Length + self.th)
		self.x = delta_S * math.cos(self.th) + self.x
		self.y = delta_S * math.sin(self.th) + self.y

		model_state_msg = ModelState()
		model_state_msg.model_name = 'duckiebot_with_gripper'
		model_state_msg.pose.position.x = self.x
		model_state_msg.pose.position.y = self.y
		model_state_msg.pose.position.z = 0

		new_quaternion = tf.transformations.quaternion_from_euler(0, 0, self.th)
		model_state_msg.pose.orientation.x = new_quaternion[0]
		model_state_msg.pose.orientation.y = new_quaternion[1]
		model_state_msg.pose.orientation.z = new_quaternion[2]
		model_state_msg.pose.orientation.w = new_quaternion[3]

		self.pub_gazebo.publish(model_state_msg)

		self.previousR = encoder_msg.x
		self.previousL = encoder_msg.y

if __name__ == "__main__":
	rospy.init_node("gazebo_publisher",anonymous=False)
	gazebo_publisher = gazebo_publisher()
	rospy.spin()
