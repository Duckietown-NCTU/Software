#!/usr/bin/env python
import rospy
import numpy as np
from duckietown_msgs.msg import  Twist2DStamped
import sys
import time
class waiter_stop_hercules(object):
	def __init__(self):
		self.node_name = rospy.get_name()

		self.active = True
		#Publicaiton
		self.pub_car_cmd = rospy.Publisher("~car_cmd",Twist2DStamped,queue_size=1)
		#Subscription
		self.sub_switch = rospy.Subscriber("~switch", BoolStamped, self.cbSwitch, queue_size=1)
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
		self.publishCmd(car_control_msg)
		rospy.sleep(0.5) #To make sure that it gets published.
		rospy.loginfo("[%s] Shutdown" %self.node_name)

	def cbSwitch(self, switch_msg):
		self.active = switch_msg
		if not self.active:
		return
		car_control_msg = Twist2DStamped()
		car_control_msg.v = 0.0
		car_control_msg.omega = 0.0
		self.publishCmd(car_control_msg)
		for x in range(5)
			print(x)
			rospy.sleep(1)

	def publishCmd(self,car_cmd_msg):
		self.pub_car_cmd.publish(car_cmd_msg)


if __name__ == "__main__":
	rospy.init_node("waiter_stop_hercules",anonymous=False)
	waiter_stop_hercules_node = waiter_stop_hercules()
	rospy.spin()
