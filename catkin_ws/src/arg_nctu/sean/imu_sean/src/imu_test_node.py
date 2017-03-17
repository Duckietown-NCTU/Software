#!/usr/bin/env python
import rospy
import time
import math
import numpy
import tf
from Adafruit_LSM303 import Adafruit_LSM303
from Gyro_L3GD20 import Gyro_L3GD20
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField

class AdafruitIMU(object):
    def __init__(self):
        self.node_name=rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))
	self.G=9.80665
	self.DEG2RAD=0.01744533
	self.signOfZ=0
        # Setup compass and accelerometer
        self.compass_accel=Adafruit_LSM303()
        # Setup gyroscope
        self.gyro=Gyro_L3GD20()
        # Setup Parameters
        self.pub_timestep=self.setupParam("~pub_timestep",0.02)
        # Publisher
        self.pub_imu=rospy.Publisher("~adafruit_imu",Imu,queue_size=10)
        self.pub_mag=rospy.Publisher("~adafruit_mag",MagneticField,queue_size=10)
        self.pub_timer=rospy.Timer(rospy.Duration.from_sec(self.pub_timestep),self.publish)
    def setupParam(self,param_name,default_value):
        value=rospy.get_param(param_name,default_value)
        # Write to parameter server for transparancy
        rospy.set_param(param_name,value)
        rospy.loginfo("[%s] %s = %s" %(self.node_name,param_name,value))
        return value
    def publish(self,event):
        compass_accel=self.compass_accel.read()
        compass=compass_accel[0]
        accel=compass_accel[1]
        gyro=self.gyro.read()
        # Put together an IMU message
        imu_msg=Imu()
        imu_msg.header.stamp=rospy.Time.now()
	imu_msg.header.frame_id="sean"
        # covariance matrix
        imu_msg.orientation_covariance[0]=-1
        imu_msg.angular_velocity_covariance[0]=-1
        imu_msg.linear_acceleration_covariance[0]=-1
        # angular velocity
        imu_msg.angular_velocity.x=gyro[0][0]*self.DEG2RAD
        imu_msg.angular_velocity.y=gyro[0][1]*self.DEG2RAD
        imu_msg.angular_velocity.z=gyro[0][2]*self.DEG2RAD
        # linear acceleration
        imu_msg.linear_acceleration.x=accel[0]*self.G
        imu_msg.linear_acceleration.y=accel[1]*self.G
        imu_msg.linear_acceleration.z=accel[2]*self.G
        # pitch roll yaw
        roll=math.atan2(imu_msg.linear_acceleration.y,imu_msg.linear_acceleration.z)
	if (imu_msg.linear_acceleration.y * math.sin(roll) + imu_msg.linear_acceleration.z * math.cos(roll) == 0):
		#pitch = imu_msg.linear_acceleration.x > 0 ? (math.pi / 2) : (math.pi / 2)
		if	imu_msg.linear_acceleration.x > 0:
			pitch=math.pi/2
		else:
			pitch=-math.pi/2
	else:
		pitch = math.atan(-imu_msg.linear_acceleration.x / (imu_msg.linear_acceleration.y * math.sin(roll) + imu_msg.linear_acceleration.z * math.cos(roll)))
	yaw = math.atan2(compass[2] * math.sin(roll) - compass[1] * math.cos(roll), compass[0] * math.cos(pitch) + compass[1] * math.sin(pitch) * math.sin(roll) + compass[2] * math.sin(pitch) * math.cos(roll))
        # orientation
        imu_msg.orientation.x=roll*180/math.pi
        imu_msg.orientation.y=pitch*180/math.pi
        imu_msg.orientation.z=yaw*180/math.pi
        imu_msg.orientation.w=0
        # publish
        self.pub_imu.publish(imu_msg)

        # Put together a magnetometer message
        mag_msg=MagneticField()
        mag_msg.header.stamp=rospy.Time.now()

        mag_msg.magnetic_field.x=compass[0]
        mag_msg.magnetic_field.y=compass[1]
        mag_msg.magnetic_field.z=compass[2]
        # publish
        self.pub_mag.publish(mag_msg)
	#def on_shutdown(self):
		#rospy.loginfo("[%s] Shutting down." %(self.node_name))
if __name__=="__main__":
	rospy.init_node("Adafruit_IMU",anonymous=False)
	adafruit_IMU=AdafruitIMU()
	#rospy.on_shutdown(adafruit_IMU.on_shutdown)
	rospy.spin()
