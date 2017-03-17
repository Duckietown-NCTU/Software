#!/usr/bin/env python
import rospy
import roscpp
import numpy as np
import math
from duckietown_msgs.msg import  Twist2DStamped
from geometry_msgs.msg import Point, Quaternion, Pose
import tf
import sys
import time
import threading
class mocap_patrol(object):
	def __init__(self):
		self.node_name = rospy.get_name()
		# vehicle position update by mocap
		self.position = Point()
		# vehicle yaw
		self.yaw = 0
		self.dist = 0
		# state switch
		self.switch = True
		# arrived label
		self.arrived = False
		# patrol mode in S type
		self.label = 0
		# target_point
		self.X = [1.04, 1.01, -0.11, -0.08, 1.02, 0.98, -0.14, -0.08, 1]
		self.Y = [0.35, 0.71, 0.58, 0.96, 0.94, 1.3, 1.2, 1.55, 1.53]
		# spining_angle
		self.R = [30 60 90 120 150 180 210 240 270 300 330 355]
		self.R_order = 0
		# Publicaiton
		self.pub_car_cmd_ = rospy.Publisher("~car_cmd",Twist2DStamped,queue_size=1)
		# Subscription
		self.sub_vehicle_pose_ = rospy.Subscriber("~vehicle_pose", Pose, self.cbPose, queue_size=1)

		# safe shutdown
		rospy.on_shutdown(self.custom_shutdown)

		# timer
		rospy.loginfo("[%s] Initialized " %(rospy.get_name()))

	def custom_shutdown(self):

		rospy.loginfo("[%s] Shutting down..." %self.node_name)

		# Send stop command
		car_control_msg = Twist2DStamped()
		car_control_msg.v = 0.0
		car_control_msg.omega = 0.0
		self.pub_car_cmd_.publish(car_cmd_msg)
		rospy.sleep(0.5) #To make sure that it gets published.
		rospy.loginfo("[%s] Shutdown" %self.node_name)
		
	def cbPose(self, pose_msg):
		# read vehicle position form mocap pose message
		self.position.x = pose_msg.position.x
		self.position.y = pose_msg.position.y
		self.position.z = pose_msg.position.z
		# convert quaternion of mocap pose message to rpy
		quaternion = (pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)
		#roll = euler[2] * 180/math.pi
		#pitch = euler[1] * 180/math.pi
		if self.switch == False or self.arrived == True:

			self.yaw  = 180 - euler[0] * 180/math.pi
			print '2_vehicle yaw: ', self.yaw

		else:
			self.yaw  = euler[0] * 180/math.pi
			if(self.yaw > 0):
				self.yaw = 180 - self.yaw
			else:
				self.yaw = -180 -self.yaw
			print '1_vehicle yaw: ', self.yaw
			
		#initial a target point for test
		target_point = self.set_target_point(self.label)
		target_yaw = self.get_yaw_two_point(self.position, target_point)
		dist = self.get_dist_two_point(self.position, target_point)

		# vehicle turn by yaw angle difference
		if self.arrived == True: 
			rotation()	
		else:
			get_target_point(target_yaw, dist)
			



	def rotation(self)

		if(abs(self.yaw - self.R[self.R_order]) >= 10):
			print 'arrived__spinning'
			self.publish_car_cmd(0.3, -((self.yaw - target_yaw) * 0.3), 0.25)
		else:
			self.R_order = self.R_order + 1

			print "the  ",self.R_order, "  rotation_point"

			if self.R_order > 11

				self.arrived == False


	def get_target_point(self, target_yaw, dist):

		if(abs(self.yaw - target_yaw) >= 10):
			print 'spinning'
			print 'omega', -((self.yaw - target_yaw) * 0.1)
			self.publish_car_cmd(0.3, -((self.yaw - target_yaw) * 0.3), 0.25)
			return

		if(dist > 0.1):
			print 'moving'
			print 'distance' , dist
			self.publish_car_cmd(0.5, 0, 0.25)
		else:
			print "**************destination arrived*****************"
			print "label = ",self.label
			self.R_order = 0

			if self.label == 1 or self.label == 5:
				self.switch = False
			elif self.label == 0 or self.label == 3 or self.label == 7:
				self.switch = True
			else:	
				print "************label no change**********"
			self.label = self.label + 1
			print "label switch to ",self.label
			self.arrived == True


	def set_target_point(self, order):
		# set a target_point
                print "the ",(order+1)," point"

		target_point = Point()
		target_point.x = self.X[order]
		target_point.y = self.Y[order]
		target_point.z = 0

		return target_point

	def get_yaw_two_point(self, source_point, target_point):
		# calculate arctan(in rad) of two point
		dx = target_point.x - source_point.x
		dy = target_point.y - source_point.y
		yaw = math.atan(dy/dx) * 180/math.pi
		#print 'original yaw', yaw
		#print 'dx', dx
		#print 'dy', dy
		# rad compensation
		if self.switch == False:
			if( dx > 0 and dy > 0):
				yaw = yaw
			elif( dx < 0):
				yaw = yaw + 180
			elif( dx > 0 and dy < 0):
				yaw = yaw + 360
		else:
			if( dx > 0):
				yaw = yaw
			elif( dx < 0 and dy > 0):
				yaw = yaw + 180
			elif( dx < 0 and dy < 0):
				yaw = yaw - 180
			elif( dx == 0 and dy == 0):
				yaw = 0
			elif( dx == 0 and dy > 0):
				yaw = 90
			elif( dx == 0 and dy < 0):
				yaw = -90 
			print 'compensation yaw: ', yaw

		return yaw

	def publish_car_cmd(self, v, omega, duration):
		# publish car command
		print 'start motion'
		print "\n"
		car_cmd_msg = Twist2DStamped()
		car_cmd_msg.v = v
		car_cmd_msg.omega = omega
		self.pub_car_cmd_.publish(car_cmd_msg)
		rospy.sleep(duration)
		# stop 1s
		car_cmd_msg.v = 0
		car_cmd_msg.omega = 0
		self.pub_car_cmd_.publish(car_cmd_msg)
		rospy.sleep(0)		
	
	def get_dist_two_point(self, source_point, target_point):

		dx = target_point.x - source_point.x
		dy = target_point.y - source_point.y
		dist = math.sqrt(dx * dx + dy * dy)
		return dist 
    
if __name__ == "__main__":
	rospy.init_node("mocap_patrol",anonymous=False)
	mocap_patrol_node = mocap_patrol()
	rospy.spin()
