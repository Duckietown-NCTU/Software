#!/usr/bin/env python
import rospy
import time
import numpy as np
import math
from Adafruit_LSM303 import Adafruit_LSM303
from Gyro_L3GD20 import Gyro_L3GD20
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point

class AdafruitIMU(object):

    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))
	
	# Physical constants
	self.G = 9.80665 # Standart gravity at sea level (should be g, but
               		 # capitalization rules due to coding practices)
	self.DEG_TO_RAD = 0.0174533 # degrees to radians

	
	#initial parameter
	self.T = self.setupParam("~pub_timestep",0.02)
	self.wGyro = 5
	self.Rest = [0,0,0]
	self.GyroRate_odd = [0,0,0]
        # Setup compass and accelerometer        
        self.compass_accel = Adafruit_LSM303()

        # Setup gyroscope
        self.gyro = Gyro_L3GD20()

        # Publications
        self.pub_imu = rospy.Publisher("~adafruit_imu_raw", Imu, queue_size=10)
        self.pub_mag = rospy.Publisher("~adafruit_mag_raw", MagneticField, queue_size=10)
	self.pub_position = rospy.Publisher("~adafruit_imu_position", Imu, queue_size=10)

        # timer
        self.pub_timer = rospy.Timer(rospy.Duration.from_sec(self.T),self.publish)

    def setupParam(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        # Write to parameter server for transparancy
        rospy.set_param(param_name, value)
        rospy.loginfo("[%s] %s = %s " % (self.node_name, param_name, value))
        return value

    def publish(self, event):
        compass_accel = self.compass_accel.read()
        compass = compass_accel[1]
        Racc = compass_accel[0]
        gyro = self.gyro.read()
        GyroRate = gyro[0]

	GyroRate = [x * self.DEG_TO_RAD for x in GyroRate]
	

	#initial
	A_odd = [0,0]
	AvgRate = [0,0]
	A = [0,0]
	Rgyro = [0,0,0]
	# normalized

	Racc_lengh = math.sqrt(Racc[0]**2+Racc[1]**2+Racc[2]**2)
	Racc_normalized = [x / Racc_lengh for x in Racc]
	A_odd[0] = math.atan2(self.Rest[1],self.Rest[2])
	A_odd[1] = math.atan2(self.Rest[0],self.Rest[2])

	for i in range(2):
		AvgRate[i] = (GyroRate[i] + self.GyroRate_odd[i])/2
		A[i] = A_odd[i] + AvgRate[i] * self.T

	Rgyro[0] = math.sin(A[1])/math.sqrt(1+((math.cos(A[1]))**2) * ((math.tan(A[0]))**2))
	Rgyro[1] = math.sin(A[0])/math.sqrt(1+((math.cos(A[0]))**2) * ((math.tan(A[1]))**2))

	if self.Rest[2] > 0:
		RzGyro_sign = 1
	else:
		RzGyro_sign = -1

	Rgyro[2] = RzGyro_sign * math.sqrt(1-Rgyro[0]**2-Rgyro[1]**2)


	for i in range(3):
		self.Rest[i] =( Racc_normalized[i] + Rgyro[i] * self.wGyro)/(1 + self.wGyro)

	position_msg = Imu()

	position_msg.linear_acceleration.x = self.Rest[0]
        position_msg.linear_acceleration.y = self.Rest[1]
        position_msg.linear_acceleration.z = self.Rest[2]
	
	position_msg.header.frame_id = "cheng"
	self.pub_position.publish(position_msg)

	
	self.GyroRate_odd = GyroRate

        # Put together a magnetometer message
	mag_msg = MagneticField()
	mag_msg.header.stamp = rospy.Time.now()
        mag_msg.magnetic_field.x = compass[0]
        mag_msg.magnetic_field.y = compass[1]
        mag_msg.magnetic_field.z = compass[2]
    
	self.pub_mag.publish(mag_msg)

if __name__ == "__main__":
    rospy.init_node("Adafruit_IMU", anonymous=False)
    adafruit_IMU = AdafruitIMU()
    rospy.spin()
