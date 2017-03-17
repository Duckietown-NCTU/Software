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
        self.pub_image_original = rospy.Publisher("~image_with_red", Image, queue_size=2)
 		###self.pub_image_mask = rospy.Publisher("~mask_with_red", Image, queue_size=1)
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


##############################################################################################
	# define the lower and upper boundaries of the "green"
	# greenLower = (29, 86, 6)
	# greenUpper = (64, 255, 255)

	# define the lower and upper boundaries of the "red"

	redLower1 = (0, 50, 90) #~
	redUpper1 = (10, 255, 255) #~
	redLower2 = (170, 50, 90) #~
	redUpper2 = (180, 255, 255) #~
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, redLower1, redUpper1) + cv2.inRange(hsv, redLower2, redUpper2) #~
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	#initialize center	
	x = 320
	y = 240 
	radius = 0
	# draw the region line
	cv2.line(image,(120,0),(120,480),(100,100,100),2)
	cv2.line(image,(320,0),(320,480),(150,150,150),2)
	cv2.line(image,(520,0),(520,480),(100,100,100),2) 

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


		# only proceed if the radius meets a minimum size
		if radius > 40:
			# draw the circle and centroid on the frame, (=image)
			# then update the list of tracked points
			
			cv2.circle(mask, (int(x), int(y)), int(radius),(255, 255, 255), 2)
			
			cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(image, center, 5, (0, 0, 255), -1)

			###cv2.line(mask,(120,0),(120,480),(100,100,100),1)
			###cv2.line(mask,(320,0),(320,480),(150,150,150),1)
			###cv2.line(mask,(520,0),(520,480),(100,100,100),1)
			#print "center: %f, %f" % (x,y)

			#h=np.size(image,0)
			#w=np.size(image,1)
			#print "h x w = %d x %d" % (h,w)
##############################################################################################
        car_control_msg = Twist2DStamped()
	
	if radius > 140:  # 1. full speed w/out following(turning)
		cv2.putText(image,"FULL SPEED!!!", (int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0),4)
		print "Running!!!~~~~~"
		car_control_msg.v=1.2
		car_control_msg.omega=0
		self.publishCmd(car_control_msg)

	elif radius > 40:  # 2. medium speed w/out following(turning left or right)
		cv2.putText(image,"following~~~", (int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0),3)
		car_control_msg.v=0.55
		if x > 320:
			if x < 520:
				car_control_msg.omega=(320-x)*0.01
				cv2.putText(image,"<2>", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
			else:
				car_control_msg.omega=-2
				cv2.putText(image,"<1>", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
		else:
			if x > 120:
				car_control_msg.omega=(320-x)*0.01
				cv2.putText(image,"<3>", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
			else:
				car_control_msg.omega=2
				cv2.putText(image,"<4>", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
		self.publishCmd(car_control_msg)

	else:  # 3. slow speed ...didn't detect any red spot(or smaller than threshold radius)
		cv2.putText(image,"hanging around...", (int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0),3)
		car_control_msg.v=0.1
		car_control_msg.omega=0
	self.publishCmd(car_control_msg)

         #rospy.Timer(rospy.Duration.from_sec(1), self.publishCmd(car_control_msg))
         #car_control_msg.v=0
         #car_control_msg.omega=0
         #self.publishCmd(car_control_msg)

        # debuging
        # self.pub_counter += 1
        # if self.pub_counter % 50 == 0:
        #     self.pub_counter = 1
        #     print "lane_controller publish"
        #     print car_control_msg
##############################################################################################        
	
	alpha = 0.7
	beta = (1.0 - alpha);
	mask_full = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
	finalImg = cv2.addWeighted(image, alpha, mask_full, beta, 0.0);


	#cv2.rectangle(image, (100, 100), (120, 140), (0, 255, 0), 2)
        #image_msg_out = self.bridge.cv2_to_imgmsg(mask, "mono8")
        image_msg_out = self.bridge.cv2_to_imgmsg(finalImg, "bgr8")
        image_msg_out.header.stamp = image_msg.header.stamp
        self.pub_image_original.publish(image_msg_out)

        ###image_msg_out = self.bridge.cv2_to_imgmsg(mask, "mono8")
        ###image_msg_out.header.stamp = image_msg.header.stamp
        ###self.pub_image_mask.publish(image_msg_out)       

if __name__ == "__main__":
    rospy.init_node("lane_controller",anonymous=False)
    lane_control_node = lane_controller()
    rospy.spin()
