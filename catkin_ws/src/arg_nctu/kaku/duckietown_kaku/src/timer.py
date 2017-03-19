#!/usr/bin/env python
import rospy
import numpy as np
from duckietown_msgs.msg import BoolStamped, FSMState
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
import time
import math

class Timer(object):
	def __init__(self):
		self.node_name = "timer"
		# fsm switch
		self.active = True
		# timer state
		self.start = False
		self.state = ""
		## publishers and subscribers
		self.sub_mode = rospy.Subscriber("~mode",FSMState, self.processStateChange)
		self.sub_in_lane = rospy.Subscriber("~in_lane", BoolStamped, self.processTimer)
		self.sub_switch = rospy.Subscriber("~switch", BoolStamped, self.cbSwitch, queue_size=1)
		self.pub_time_is_up = rospy.Publisher("~time_is_up", BoolStamped, queue_size=1, latch=True)

	def processStateChange(self, msg):
		self.state=msg.state #

	# def processTimer(self, msgg):
	# 	if self.active: # fsm switch on
	# 		if (self.state == "LANE_FOLLOWING_TURN_RIGHT") or (self.state == "LANE_FOLLOWING_TURN_LEFT"): # if in fsm state "left turn" or "right turn""
	# 			if not self.start: # if timer not start yet
	# 				self.timer_start = time.time() # record start time
	# 				self.start = True # change timer state to start
	# 				print "start time: ", self.timer_start
	# 			self.timer_end = time.time() # record time now
	# 			print "end time: ", self.timer_end
	# 			if (self.timer_end - self.timer_start) > 4: #if time duration between start time and time now bigger than 2 seconsds
	# 				# publish time is up
	# 				msg = BoolStamped()
	# 				msg.data = True
	# 				self.pub_time_is_up.publish(msg)
	# 				print "time is up"
	# 	if not self.active: # fsm switch off
	# 		# publish timer is off and reset timer state to not starting
	# 		msg = BoolStamped()
	# 		msg.data = False
	# 		self.pub_time_is_up.publish(msg)
	# 		self.start = False

	def processTimer(self, msgg):
		if self.active: # fsm switch on
			if not self.start: # if timer not start yet
				self.timer_start = time.time() # record start time
				self.start = True # change timer state to start
				print "start time: ", self.timer_start
			self.timer_end = time.time() # record time now
			print "end time: ", self.timer_end
			if (self.timer_end - self.timer_start) > 1: #if time duration between start time and time now bigger than 2 seconsds
				# publish time is up
				msg = BoolStamped()
				msg.data = True
				self.pub_time_is_up.publish(msg)
				print "time is up"
		if not self.active: # fsm switch off
			# publish timer is off and reset timer state to not starting
			msg = BoolStamped()
			msg.data = False
			self.pub_time_is_up.publish(msg)
			self.start = False

	def cbSwitch(self, switch_msg):
		self.active = switch_msg.data

	def onShutdown(self):
		rospy.loginfo("[Timer] Shutdown.")

if __name__ == '__main__': 
	rospy.init_node('Timer',anonymous=False)
	timer_node = Timer()
	rospy.on_shutdown(timer_node.onShutdown)
	rospy.spin()

