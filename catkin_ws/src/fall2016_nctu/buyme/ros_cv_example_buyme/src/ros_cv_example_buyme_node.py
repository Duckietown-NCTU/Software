#!/usr/bin/env python
import rospy
import numpy as np
import math
from duckietown_msgs.msg import  Twist2DStamped, LanePose
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
from duckietown_utils.jpg import image_cv_from_jpg
class ros_cv_example_wama(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.lane_reading = None
        self.pub_counter = 0

        # Setup parameters
        self.setGains()

        #cv
        self.bridge = CvBridge()
          
        # Publicaiton
        self.pub_image_original = rospy.Publisher("~image_ros_cv", Image, queue_size=1)
        #self.pub_image_gray = rospy.Publisher("~image_gray", Image, queue_size=1)
        # Subscriptions
        self.sub_lane_reading = rospy.Subscriber("~image", CompressedImage, self.cbPose, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)

        # timer
        self.gains_timer = rospy.Timer(rospy.Duration.from_sec(1.0), self.getGains_event)
        rospy.loginfo("[%s] Initialized " %(rospy.get_name()))

    def setupParameter(self,param_name,default_value):
        value = rospy.get_param(param_name,default_value)
        rospy.set_param(param_name,value) #Write to parameter server for transparancy
        rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value

    def setGains(self):
        v_bar = 0.5 # nominal speed, 0.5m/s
        k_theta = -2.0
        k_d = - (k_theta ** 2) / ( 4.0 * v_bar)
        theta_thres = math.pi / 6
        d_thres = math.fabs(k_theta / k_d) * theta_thres
        d_offset = 0.0

        self.v_bar = self.setupParameter("~v_bar",v_bar) # Linear velocity
        self.k_d = self.setupParameter("~k_d",k_theta) # P gain for theta
        self.k_theta = self.setupParameter("~k_theta",k_d) # P gain for d
        self.d_thres = self.setupParameter("~d_thres",theta_thres) # Cap for error in d
        self.theta_thres = self.setupParameter("~theta_thres",d_thres) # Maximum desire theta
        self.d_offset = self.setupParameter("~d_offset",d_offset) # a configurable offset from the lane position

    def getGains_event(self, event):
        v_bar = rospy.get_param("~v_bar")
        k_d = rospy.get_param("~k_d")
        k_theta = rospy.get_param("~k_theta")
        d_thres = rospy.get_param("~d_thres")
        theta_thres = rospy.get_param("~theta_thres")
        theta_thres = rospy.get_param("~theta_thres")
        d_offset = rospy.get_param("~d_offset")

        params_old = (self.v_bar,self.k_d,self.k_theta,self.d_thres,self.theta_thres, self.d_offset)
        params_new = (v_bar,k_d,k_theta,d_thres,theta_thres, d_offset)

        if params_old != params_new:
            rospy.loginfo("[%s] Gains changed." %(self.node_name))
            rospy.loginfo("old gains, v_var %f, k_d %f, k_theta %f, theta_thres %f, d_thres %f, d_offset %f" %(params_old))
            rospy.loginfo("new gains, v_var %f, k_d %f, k_theta %f, theta_thres %f, d_thres %f, d_offset %f" %(params_new))
            self.v_bar = v_bar
            self.k_d = k_d
            self.k_theta = k_theta
            self.d_thres = d_thres
            self.theta_thres = theta_thres
            self.d_offset = d_offset

    
    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." %self.node_name)
        
        # Stop listening
        self.sub_lane_reading.unregister()

        # Send stop command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0.0
        car_control_msg.omega = 0.0
        rospy.sleep(0.5) #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)



    def cbPose(self,image_msg):
        
        narr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(narr, cv2.CV_LOAD_IMAGE_COLOR)
        
        
        cv2.rectangle(image, (100, 100), (120, 140), (0, 255, 0), 2)
       
        image_msg_out = self.bridge.cv2_to_imgmsg(image, "bgr8")
        image_msg_out.header.stamp = image_msg.header.stamp
        self.pub_image_original.publish(image_msg_out)

if __name__ == "__main__":
    rospy.init_node("ros_cv_example_wama",anonymous=False)
    ros_cv_example_wama_node = ros_cv_example_wama()
    rospy.spin()
