#!/usr/bin/env python
import rospy
import numpy as np
import math
from duckietown_msgs.msg import  Twist2DStamped, LanePose, AprilTags
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
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
        self.sub_lane_reading = rospy.Subscriber("/alphaduck/apriltags_fast_node/apriltags", AprilTags, self.aprilProcess, queue_size=1)

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
    
    def aprilProcess(self, apriltags_msg):
 #       self.status.received()
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0.0
        car_control_msg.omega = 0.0

        self.publishCmd(car_control_msg)
        if len(apriltags_msg.detections) == 0:
            rospy.sleep(0.04)
            return
        
        (x, y) = apriltags_msg.detections[0].cxy
        print(x) 
        if x < 280.0: 
            car_control_msg.v = 0.15
            car_control_msg.omega = 0.2
            print("turn left")
        elif x > 360.0:
            car_control_msg.v = 0.0
            car_control_msg.omega = -0.5
            print("turn right")
        else:
            car_control_msg.v = 0.3 
            car_control_msg.omega = 0.0
            print("fuck")
        self.publishCmd(car_control_msg)
        if not self.active:
              return
        #rospy.sleep(0.4) 
        #thread = threading.Thread(target=self.processImage,args=(image_msg,))
        #thread.setDaemon(True)
        #thread.start()

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
        
         #       image_cv = cv2.imdecode(np.fromstring(image_msg.data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        #image_cv = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        
        #image = image_cv_from_jpg(image_msg.data)
        faceCascade = cv2.CascadeClassifier('/home/ubuntu/duckietown/catkin_ws/src/spring2016_nctu/wama/face_detector/src/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray,scaleFactor=5,minNeighbors=5,minSize=(10, 10),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        print "Found {0} faces!".format(len(faces))
        
        for (x, y, w, h) in faces:
           cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #   cv2.imshow("preview", image)
        #   cv2.waitKey(0)
        #   cv2.destroyAllWindows()
        
        
        #cv2.rectangle(image, (100, 100), (120, 140), (0, 255, 0), 2)
        image_msg_out = self.bridge.cv2_to_imgmsg(image, "bgr8")
        image_msg_out.header.stamp = image_msg.header.stamp
        self.pub_image_original.publish(image_msg_out)
        #image_msg_out = self.bridge.cv2_to_imgmsg(gray, "bgr8")
        #image_msg_out.header.stamp = image_msg.header.stamp
        #self.pub_image_gray.publish(image_msg_out)
        #face_cascade = cv2.CascadeClassifier('~/haarcascade_frontalface_default.xml')
        #eye_cascade = cv2.CascadeClassifier('~/haarcascade_eye.xml')
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #for (x,y,w,h) in faces:
        #    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #    roi_gray = gray[y:y+h, x:x+w]
        #    roi_color = img[y:y+h, x:x+w]
        #    self.eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in self.eyes:
        #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #    cv2.imwrite('hard.png',img)
        #    cv2.destroyAllWindows()


        car_control_msg = Twist2DStamped()
        #car_control_msg.header = lane_pose_msg.header
        #car_control_msg.v = self.v_bar #*self.speed_gain #Left stick V-axis. Up is positive
        
        #if math.fabs(cross_track_err) > self.d_thres:
        #    cross_track_err = cross_track_err / math.fabs(cross_track_err) * self.d_thres
        #car_control_msg.omega =  self.k_d * cross_track_err + self.k_theta * heading_err #*self.steer_gain #Right stick H-axis. Right is negative
        
        # controller mapping issue
        # car_control_msg.steering = -car_control_msg.steering
        # print "controls: speed %f, steering %f" % (car_control_msg.speed, car_control_msg.steering)
        # self.pub_.publish(car_control_msg)
	if len(faces) != 0:
         car_control_msg.v=0
         car_control_msg.omega=0
         self.publishCmd(car_control_msg)

        if len(faces) == 0:
         car_control_msg.v=0.5
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

if __name__ == "__main__":
    rospy.init_node("lane_controller",anonymous=False)
    lane_control_node = lane_controller()
    rospy.spin()
