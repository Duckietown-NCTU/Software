#!/usr/bin/env python
import rospy
import numpy as np
from duckietown_msgs.msg import BoolStamped, FSMState
from std_msgs.msg import Float32
from geometry_msgs.msg import Point, Twist
import time
import math
from Adafruit_MotorHAT import Adafruit_MotorHAT
from sensor_msgs.msg import Joy

class dc_grab(object):
    def __init__(self):
	    self.node_name = "dc_grab"
	    self.active = True
	    self.motorhat = Adafruit_MotorHAT(addr=0x60)
	    self.dc_grab = self.motorhat.getMotor(3)

	    self.sub_joy_ = rospy.Subscriber("joy", Joy, self.cbJoy, queue_size=1)
	    self.pub_gazebo_grab = rospy.Publisher('/duckiebot_with_gripper/gripper_cmd_vel', Twist, queue_size=1)

    def cbJoy(self, msg):
		self.joy = msg
		self.processButtons(msg)

    def processButtons(self, msg):
		grab_state_msg = Twist()
		grab_state_msg.linear.x = 0
		grab_state_msg.linear.y = 0
		grab_state_msg.linear.z = 0

		grab_state_msg.angular.x = 0
		grab_state_msg.angular.y = 0
		grab_state_msg.angular.z = 0

		if (self.joy.buttons[0] == 1):

			self.dc_grab.setSpeed(200)
			self.dc_grab.run(Adafruit_MotorHAT.BACKWARD)

			grab_state_msg.angular.z = -1
			self.pub_gazebo_grab.publish(grab_state_msg)

		if (self.joy.buttons[1] == 1):

			self.dc_grab.setSpeed(200)
			self.dc_grab.run(Adafruit_MotorHAT.FORWARD)

			grab_state_msg.angular.z = +1
			self.pub_gazebo_grab.publish(grab_state_msg)

		if (self.joy.buttons[2] == 1):

			self.dc_grab.run(Adafruit_MotorHAT.RELEASE)

 
    def onShutdown(self):
        rospy.loginfo("[dc_grab] Shutdown.")

if __name__ == '__main__': 
    rospy.init_node('dc_grab',anonymous=False)
    dc_grab = dc_grab()
    rospy.on_shutdown(dc_grab.onShutdown)
    rospy.spin()

