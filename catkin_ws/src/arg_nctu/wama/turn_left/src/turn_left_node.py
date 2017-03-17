#!/usr/bin/env python
import rospy
import numpy as np
import math
import time
from duckietown_msgs.msg import  Twist2DStamped, BoolStamped
class TurnLeftNode(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.active = False

        # Publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)

       	self.pub_back_lane_following= rospy.Publisher("~back_lane_following", BoolStamped, queue_size=1)

        # Subscribers
        self.sub_switch = rospy.Subscriber("~switch", BoolStamped, self.cbSwitch, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)

        rospy.loginfo("[%s] Initialized " %(rospy.get_name()))


    def cbSwitch(self, switch_msg):
    	self.active = switch_msg.data

    	if not self.active:
    		back_lane_following_msg = BoolStamped()
        	back_lane_following_msg = False
        	self.pub_back_lane_following.publish(back_lane_following_msg)


    	if self.active:
    		car_control_msg = Twist2DStamped()
        	car_control_msg.v = 1.0
        	car_control_msg.omega = 3.0
       		self.publishCmd(car_control_msg)
       		rospy.sleep(3)
       		car_control_msg.v = 0.0
        	car_control_msg.omega = 0.0
        	self.publishCmd(car_control_msg)

        	back_lane_following_msg = BoolStamped()
        	back_lane_following_msg = True
        	self.pub_back_lane_following.publish(back_lane_following_msg)


    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." %self.node_name)
   
        # Send stop command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0.0
        car_control_msg.omega = 0.0
        self.publishCmd(car_control_msg)
        rospy.sleep(0.5) #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)


    def publishCmd(self,car_cmd_msg):
		self.pub_car_cmd.publish(car_cmd_msg)

if __name__ == "__main__":
    rospy.init_node('turn_left',anonymous=False)
    turn_left_node = TurnLeftNode()
    rospy.spin()
