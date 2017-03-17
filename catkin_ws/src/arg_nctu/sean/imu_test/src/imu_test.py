#!/usr/bin/env python
import rospy
import time
from math import atan2
from math import pi as PI
import numpy as np
from Adafruit_LSM303 import Adafruit_LSM303
from Gyro_L3GD20 import Gyro_L3GD20
#from imu_test.msg import angular
class IMU_test(object):
	def __init__(self):
		self.node_name=rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.compass_accel=Adafruit_LSM303()
		self.gyro=Gyro_L3GD20()
		self.pub_timestep=self.setupParam("~pub_timestep",0.2)
		#self.pub_imu=rospy.Publisher("~adafruit_imu",float,queue_size=5)
		self.pub_timer=rospy.Timer(rospy.Duration.from_sec(self.pub_timestep),self.publish)
	def setupParam(self,param_name,default_value):
		value=rospy.get_param(param_name,default_value)
		rospy.set_param(param_name,value)
		rospy.loginfo("[%s] %s = %s" %(self.node_name),param_name,value)
		return value
	def publish(self,event):
		compass_accel=self.compass_accel_read()
		compass=compass_accel[0]
		accel=compass_accel[1]
		angular_msg=float64()
		t_roll=accel[0]*accel[0]+accel[2]*accel[2]
		#angular_msg.roll=atan2(accel[1],sqrt(t_roll))*180/PI
		roll=atan2(accel[1],sqrt(t_roll))*180/PI
		t_pitch=accel[1]*accel[1]+accel[2]*accel[2]
		#angular_msg.pitch=atan2(accel[0].t_pitch)*180/PI
		pitch=atan2(accel[0],t_pitch)*180-PI
		#self.pub_imu.publish(angular_msg)
		rospy.loginfo("[%s] roll %s  pitch %s"%(self.node_name),roll,pitch)
if __name__=="__main__":
	rospy.init_node("Adafruit_IMU",anonymous=False)
	adafruit_IMU=AdafruitIMU()
	rospy.spin()
