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

class camera_mount(object):
    def __init__(self):
	    self.node_name = "camera_mount"
	    self.active = True

	    self.pwm = Adafruit_PCA9685.PCA9685()
	    # self.servo_min = 340  # Min pulse length out of 4096
	    # self.servo_max = 600  # Max pulse length out of 4096
	    self.horL = 485
	    self.ver = 475
	    self.horR = 240
	    self.pwm.set_pwm_freq(60)

	    self.sub_joy_ = rospy.Subscriber("joy", Joy, self.cbJoy, queue_size=1)

    def cbJoy(self, msg):
	    self.joy = msg
	    self.processButtons(msg)

    def processButtons(self, msg):
		if (self.joy.buttons[0] == 1):
				self.horL +=40
				print " horL = ",self.horL
				self.horR -=40
				print " horR = ",self.horR
				self.pwm.set_pwm(2, 0, self.horR)
				self.pwm.set_pwm(10, 0, self.horL)

		if (self.joy.buttons[2] == 1):
                                self.horL -=40
                                print " horL = ",self.horL
                                self.horR +=40
                                print " horR = ",self.horR
                                self.pwm.set_pwm(2, 0, self.horR)
                                self.pwm.set_pwm(10, 0, self.horL)

		if (self.joy.buttons[1] == 1):
				self.ver +=10
				print " ver = ",self.ver
				self.pwm.set_pwm(0, 0, self.ver)
				self.pwm.set_pwm(4, 0, self.ver)
				self.pwm.set_pwm(8, 0, self.ver)

		if (self.joy.buttons[3] == 1):
				self.ver -=10
				print " ver = ",self.ver
				self.pwm.set_pwm(0, 0, self.ver)
				self.pwm.set_pwm(4, 0, self.ver)
				self.pwm.set_pwm(8, 0, self.ver)
		
    def onShutdown(self):
        rospy.loginfo("[camera_mount] Shutdown.")

if __name__ == '__main__': 
    rospy.init_node('camera_mount',anonymous=False)
    camera_mount = camera_mount()
    rospy.on_shutdown(camera_mount.onShutdown)
    rospy.spin()

