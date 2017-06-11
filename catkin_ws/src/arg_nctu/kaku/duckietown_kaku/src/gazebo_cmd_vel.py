#!/usr/bin/env python
import rospy
import roscpp
import numpy as np
import math
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped,Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Int64
import tf
import sys
import time
import tf2_ros
class gazebo_cmd_vel(object):
	def __init__(self):
		self.node_name = rospy.get_name() 
		self.pretime = 0.0
		self.x = 0
		self.y = 0
		self.th = 0
		self.d_PerCount = ((3.14159265 * 0.13) / 5940)*0.3/0.635
		self.Length =  0.18
		self.previousR = 0
		self.previousL = 0

		self.pub_gazebo = rospy.Publisher('/duckiebot_with_gripper/cmd_vel', Twist, queue_size=1)
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
		time_now = time.time()
		delta_R = (encoder_msg.x - self.previousR) * self.d_PerCount
		delta_L= (encoder_msg.y - self.previousL) * self.d_PerCount

		delta_S = (delta_R + delta_L)/2
		th = (delta_R - delta_L)/self.Length


		time_inteval = time_now - self.pretime
		# time_inteval = 1/30

		if self.pretime != 0:
			model_state_msg = Twist()
			model_state_msg.linear.x = delta_S *(1/time_inteval)
			model_state_msg.linear.y = 0
			model_state_msg.linear.z = 0

			model_state_msg.angular.x = 0
			model_state_msg.angular.y = 0
			model_state_msg.angular.z = th *(1/time_inteval)
			
			self.pub_gazebo.publish(model_state_msg)

		self.previousR = encoder_msg.x
		self.previousL = encoder_msg.y
		self.pretime = time_now

if __name__ == "__main__":
	rospy.init_node("gazebo_cmd_vel",anonymous=False)
	gazebo_cmd_vel = gazebo_cmd_vel()
	rospy.spin()
