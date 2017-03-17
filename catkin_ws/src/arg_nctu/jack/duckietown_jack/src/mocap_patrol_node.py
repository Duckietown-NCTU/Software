#!/usr/bin/env python
import rospy
import roscpp
import numpy as np
import math
from duckietown_msgs.msg import  Twist2DStamped, BoolStamped
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
		# patrol mode in S type
		self.label = 0
		# start patrolling
		self.start = False
		# reset target
		self.reset = False

		#self.X = [1.07, 0.98, -0.17, -0.136, 1.0, 0.92, -0.18, -0.14, 0.98, 1.03, 0.68, 0.65, 0.03, 0.49, 0.12, 0.1, -0.194, -0.07 ]
		#self.X = [1.45, 1.38, 0.10, 0.15, 1.45, 1.37, 0.08, 0.13, 1.48]
		#self.Y = [0.16, 0.56, 0.44, 0.80, 0.73, 1.12, 1.04, 1.38, 1.34]
		#self.Y = [0.35, 0.71, 0.56, 0.942, 0.96, 1.3, 1.19, 1.54, 1.56, 0.29, 0.315, 1.56, 1.5, 0.245, 0.29, 1.56, 1.5, 0.26 ]

		#self.X = [1.36, 1.36, 0.16, 0.16, 1.36, 1.36, 0.16, 0.16, 1.36, 1.36, 1.06, 1.06, 0.76, 0.76, 0.46, 0.46, 0.16, 0.16]
		#self.Y = [0.16, 0.46, 0.46, 0.76, 0.76, 1.06, 1.06, 1.36, 1.36, 0.16, 0.16, 1.36, 1.36, 0.16, 0.16, 1.36, 1.36, 0.16]

		self.X = [1.41, 1.35, 0.09, 0.17, 1.41, 1.36, 0.09, 0.16, 1.44, 1.31, 0.94, 1.08, 0.68, 0.71, 0.35, 0.44, 0.10, 0.11]
		self.Y = [0.20, 0.56, 0.49, 0.86, 0.74, 1.15, 1.08, 1.41, 1.34, 0.13, 0.17, 1.41, 1.34, 0.11, 0.16, 1.42, 1.33, 0.12]


		self.yaw_target = 0
		self.yaw_old = 0
		self.yaw_new = 0
		self.kd = 0.04
		#self.kd = 0.1
		#self.kp = 0.1
		self.kp = 0.08
		

		# Publicaiton
		self.pub_car_cmd_ = rospy.Publisher("~car_cmd",Twist2DStamped,queue_size=1)
		# Subscription
		self.sub_vehicle_pose_ = rospy.Subscriber("~vehicle_pose", Pose, self.cbPose, queue_size=1)
		self.sub_start_cmd_ = rospy.Subscriber("~start_cmd", BoolStamped, self.cbStart, queue_size=1)
		self.sub_reset_cmd_ = rospy.Subscriber("~reset_cmd", BoolStamped, self.cbReset, queue_size=1)

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
		self.pub_car_cmd_.publish(car_control_msg)
		rospy.sleep(3) #To make sure that it gets published.
		rospy.loginfo("[%s] Shutdown" %self.node_name)
		
	def cbPose(self, pose_msg):
		if not(self.start):
			return
		# read vehicle position form mocap pose message
		self.position.x = pose_msg.position.x
		self.position.y = pose_msg.position.y
		self.position.z = pose_msg.position.z
		# convert quaternion of mocap pose message to rpy
		quaternion = (pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)
		#roll = euler[2] * 180/math.pi
		#pitch = euler[1] * 180/math.pi
		if self.switch == True:
			self.yaw  = euler[0] * 180/math.pi
			if(self.yaw > 0):
				self.yaw = 180 - self.yaw
			else:
				self.yaw = -180 -self.yaw
			print '1_vehicle yaw: ', self.yaw

		else:
			self.yaw  = 180 - euler[0] * 180/math.pi
			print '2_vehicle yaw: ', self.yaw
			
		#initial a target point for test
		target_point = self.set_target_point(self.label)
		target_yaw = self.get_yaw_two_point(self.position, target_point)
		dist = self.get_dist_two_point(self.position, target_point)

		# vehicle turn by yaw angle difference 
		if self.label == 18:
			print "------- patrol mode END -------"
			self.publish_car_cmd(0, 0, 0.5)
			rospy.sleep(3)
			return

		#if(abs(self.yaw - target_yaw) >= 20):
		if(True):
			
			self.yaw_target = target_yaw
			self.yaw_old = self.yaw_new
			self.yaw_new = self.yaw
			ess = self.yaw_new - self.yaw_target
			diff = self.yaw_new - self.yaw_old
			u = self.kp * ess + self.kd * diff
			print 'omega pd: ', -u
			if( u < -7):
				u = -7
			if( u > 7):
				u = 7
			print 'mega pd com: ', -u
			self.publish_car_cmd(0.3, -u , 0.2)
		if(dist > 0.05):
			print 'distance' , dist
			#self.publish_car_cmd(0.5, 0, 0.25)
			#self.publish_car_cmd(0.2, 0, 0.25)
		else:
			print "**************destination arrived*****************"
			print "label = ",self.label
			if self.label == 1 or self.label == 5 or self.label == 8:
				self.switch = False
			elif self.label == 0 or self.label == 3 or self.label == 7:
				self.switch = True
			else:	
				print "************label no change**********"
			self.label = self.label + 1
			if self.label > 17:
				self.label = 18
			print "label switch to ",self.label
	

	def set_target_point(self, order):
		# set a target_point
                print "the ",(order+1)," point"

		target_point = Point()
		target_point.x = self.X[order]
		target_point.y = self.Y[order]
		target_point.z = 0

		return target_point

	def cbStart(self, start_msg):
		self.start = start_msg.data

	def cbReset(self, reset_msg):
		if(reset_msg.data):
			self.label = 0

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

#	def publish_car_cmd_straight(self):
                # publish car command
#                print 'start motion'
#                print "\n"
#                car_cmd_msg = Twist2DStamped()
#                car_cmd_msg.v = 0.5
#                car_cmd_msg.omega = 0
#                self.pub_car_cmd_.publish(car_cmd_msg)

	
	def get_dist_two_point(self, source_point, target_point):

		dx = target_point.x - source_point.x
		dy = target_point.y - source_point.y
		dist = math.sqrt(dx * dx + dy * dy)
		return dist 
    
if __name__ == "__main__":
	rospy.init_node("mocap_patrol",anonymous=False)
	mocap_patrol_node = mocap_patrol()
	rospy.spin()
