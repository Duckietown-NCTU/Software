#!/usr/bin/env python
import rospy
import numpy as np
from duckietown_msgs.msg import BoolStamped, FSMState
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
import time
import math
import Adafruit_PCA9685
from sensor_msgs.msg import Joy

class grab(object):
    def __init__(self):
        self.node_name = "grab"
        self.active = True
	self.i = 0
	self.a = 0

	self.pwm = Adafruit_PCA9685.PCA9685()
	self.servo_min = 340  # Min pulse length out of 4096
	self.servo_max = 600  # Max pulse length out of 4096
        self.pwm.set_pwm_freq(60)

	self.sub_joy_ = rospy.Subscriber("joy", Joy, self.cbJoy, queue_size=1)

    def cbJoy(self, msg):
	self.joy = msg
	self.processButtons(msg)

    def processButtons(self, msg):
	if (self.joy.buttons[0] == 1):
		self.i = self.i + 1
		self.a = self.i % 2
	if self.a == 1:
		self.pwm.set_pwm(0, 0, self.servo_min)
	else:
		self.pwm.set_pwm(0, 0, self.servo_max)
		

 
    def onShutdown(self):
        rospy.loginfo("[grab] Shutdown.")

if __name__ == '__main__': 
    rospy.init_node('grab',anonymous=False)
    grab = grab()
    rospy.on_shutdown(grab.onShutdown)
    rospy.spin()

