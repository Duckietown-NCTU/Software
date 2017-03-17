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
import time
import threading
class lane_controller(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.lane_reading = None     
        self.pub_counter = 0
      
        self.thread_lock = threading.Lock()
        self.active = True
#        self.stats = Stats()

        #cv
        self.bridge = CvBridge()
          
        # Publicaiton
        self.pub_car_cmd = rospy.Publisher("~car_cmd",Twist2DStamped,queue_size=1)
        self.pub_image_original = rospy.Publisher("~image_with_duck", Image, queue_size=2)
 		###self.pub_image_mask = rospy.Publisher("~mask_with_duck", Image, queue_size=1)
        self.sub_lane_reading = rospy.Subscriber("~image", CompressedImage, self.cbImage, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)

        # timer
        rospy.loginfo("[%s] Initialized " %(rospy.get_name()))

    
    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." %self.node_name)
        
        # Stop listening
        self.sub_lane_reading.unregister()

        # Send stop command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0.0
        car_control_msg.omega = 0.0
        self.publishCmd(car_control_msg)
        rospy.sleep(0.5) #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)
    
    def cbImage(self, image_msg):
 #       self.status.received()
        if not self.active:
              return
       
        thread = threading.Thread(target=self.processImage,args=(image_msg,))
        thread.setDaemon(True)
        thread.start()

    def processImage(self, image_msg):
        if not self.thread_lock.acquire(False):
  #           self.status.skipped()
             return

        try:
            self.cbPose(image_msg)
        finally:
            self.thread_lock.release()

    def publishCmd(self,car_cmd_msg):
        self.pub_car_cmd.publish(car_cmd_msg)

   
    def cbPose(self, image_msg):
   #     self.stats.processed()
        #self.lane_reading = lane_pose_msg 

        #cross_track_err = lane_pose_msg.d - self.d_offset
        #heading_err = lane_pose_msg.phi

        #img = self.bridge.imgmsg_to_cv2(image_msg)
        
        narr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(narr, cv2.CV_LOAD_IMAGE_COLOR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	Cascade = cv2.CascadeClassifier("/home/ubuntu/duckietown/catkin_ws/src/spring2016_nctu/phnix/ducks_are_monsters/src/eye_classifier.xml")
	
	bananas = Cascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30, 30),
	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)
	for (x1, y1, w1, h1) in bananas:
		cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)


        car_control_msg = Twist2DStamped()
	
	if len(bananas) is not 0 :  # 1. full speed w/out following(turning)
		print "Running!!!~~~~~ %s" % len(bananas)
		
		car_control_msg.v=1.2
		car_control_msg.omega=0
		self.publishCmd(car_control_msg)

	

	else:  # 3. slow speed ...didn't detect any red spot(or smaller than threshold radius)
		print "stop"
		car_control_msg.v=0.0
		car_control_msg.omega=0
		self.publishCmd(car_control_msg)

         
##############################################################################################        
	


        image_msg_out = self.bridge.cv2_to_imgmsg(image, "bgr8")
        image_msg_out.header.stamp = image_msg.header.stamp
        self.pub_image_original.publish(image_msg_out)

            

if __name__ == "__main__":
    rospy.init_node("lane_controller",anonymous=False)
    lane_control_node = lane_controller()
    rospy.spin()
