#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import ObstacleImageDetection, ObstacleImageDetectionList, ObstacleType, Rect, BoolStamped
import sys
import threading
from rgb_led import *


class Matcher:
    CONE = [np.array(x, np.uint8) for x in [[0,80,80], [22, 255,255]] ]
    #DUCK = [np.array(x, np.uint8) for x in [[25,100,150], [35, 255, 255]] ]
    #DUCK = [np.array(x, np.uint8) for x in [[0,0,95], [50, 100, 250]] ]
    DUCK = [np.array(x, np.uint8) for x in [[0,0,95], [50, 100, 250]] ]
    terms = {ObstacleType.CONE :"cone", ObstacleType.DUCKIE:"duck"}
    def __init__(self):
        self.cone_color_low = self.setupParam("~cone_low", [0,80,80])
        self.cone_color_high = self.setupParam("~cone_high", [22, 255,255])
        self.duckie_color_low = self.setupParam("~duckie_low", [0, 0, 95])
        self.duckie_color_high = self.setupParam("~duckie_high", [50, 100,250])
        self.CONE = [np.array(x, np.uint8) for x in [self.cone_color_low, self.cone_color_high] ]
        self.DUCK = [np.array(x, np.uint8) for x in [self.duckie_color_low, self.duckie_color_high] ]

    def setupParam(self, param_name, default_value):
        value = rospy.get_param(param_name,default_value)
        rospy.set_param(param_name,value) #Write to parameter server for transparancy
        rospy.loginfo("[%s] %s = %s " %('static_object_detector_node',param_name,value))
        return value

    def get_filtered_contours(self,img, contour_type):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        if contour_type == "CONE":
            frame_threshed = cv2.inRange(hsv_img, self.CONE[0], self.CONE[1])
            ret,thresh = cv2.threshold(frame_threshed,22,255,0)
        elif contour_type == "DUCK_COLOR":
            frame_threshed = cv2.inRange(hsv_img, self.DUCK[0], self.DUCK[1])
            ret,thresh = cv2.threshold(frame_threshed,30,255,0)
        elif contour_type == "DUCK_CANNY":
            frame_threshed = cv2.inRange(hsv_img, self.DUCK[0], self.DUCK[1])
            frame_threshed = cv2.adaptiveThreshold(frame_threshed,255,\
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)
            thresh = cv2.Canny(frame_threshed, 100,200)
        else:
            return 
        
        filtered_contours = []
        
        contours, hierarchy = cv2.findContours(\
                thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        contour_area = [ (cv2.contourArea(c), (c) ) for c in contours]
        contour_area = sorted(contour_area,reverse=True, key=lambda x: x[0])

        height,width = img.shape[:2]
        print 'start box:'
        for (area,(cnt)) in contour_area:
        # plot box around contour
            x,y,w,h = cv2.boundingRect(cnt)
            box = (x,y,w,h)
            d =  0.5*(x-width/2)**2 + (y-height)**2 
            if not(h>15 and w >10 and h<200 and w<200 and d < 120000):
                    continue
            if contour_type == "DUCK_CANNY":
                continue
            if contour_type =="DUCK_COLOR": # extra filtering to remove lines
                #if not(h>25 and w>25):
                    #continue
                #if d>90000:
                    #if not(h>35 and w>35):
                        #continue
                if cv2.contourArea(cnt)==0:
                    continue
                #val = cv2.arcLength(cnt,True)**2/ cv2.contourArea(cnt)
                #if val > 35: continue
                #rect = cv2.minAreaRect(cnt)
                #ctr, sides, deg = rect
                #val  = 0.5*cv2.arcLength(cnt,True) / (w**2+h**2)**0.5
                #if val < 1.12: continue
                # add by arg
                ratio = w / h
                if ( abs(1 - ratio) >= 0.1):
                    continue#
                #if area > 1000: continue
                if (w*h <= 900):
                    continue
                print x,y,w,h
            mask = np.zeros(thresh.shape,np.uint8)
            cv2.drawContours(mask,[cnt],0,255,-1)
            mean_val = cv2.mean(img,mask = mask)
            aspect_ratio = float(w)/h
            filtered_contours.append( (cnt, box, d, aspect_ratio, mean_val) )
        return filtered_contours


    def contour_match(self, img):
        '''
        Returns 1. Image with bounding boxes added
                2. an ObstacleImageDetectionList
        '''

        object_list = ObstacleImageDetectionList()
        object_list.list = []

        height,width = img.shape[:2]
        object_list.imwidth = width
        object_list.imheight = height
        
        # get filtered contours
        cone_contours = self.get_filtered_contours(img, "CONE")
        duck_contours = self.get_filtered_contours(img, "DUCK_COLOR")

        all_contours = [duck_contours, cone_contours]
        for i, contours in enumerate(all_contours):
            for (cnt, box, ds, aspect_ratio, mean_color)  in contours:
                            
                # plot box around contour
                x,y,w,h = box
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,self.terms[i], (x,y), font, 0.5,mean_color,4)
                cv2.rectangle(img,(x,y),(x+w,y+h), mean_color,2)
                
                r = Rect()
                r.x = x
                r.y = y
                r.w = w
                r.h = h

                t = ObstacleType()
                t.type = i

                d = ObstacleImageDetection()
                d.bounding_box = r
                d.type = t
                object_list.list.append(d);
        return img, object_list

class StaticObjectDetectorNode:
    def __init__(self):
        self.name = 'static_object_detector_node'
        
        self.tm = Matcher()
        self.active = True
        self.thread_lock = threading.Lock()
        self.sub_image = rospy.Subscriber("~image_raw", Image, self.cbImage, queue_size=1)
        self.sub_switch = rospy.Subscriber("~switch",BoolStamped, self.cbSwitch, queue_size=1)
        self.pub_image = rospy.Publisher("~cone_detection_image", Image, queue_size=1)
        self.pub_detections_list = rospy.Publisher("~detection_list", ObstacleImageDetectionList, queue_size=1)
        self.bridge = CvBridge()
	turn_off_LEDs(speed=5)

        rospy.loginfo("[%s] Initialized." %(self.name))

    def cbSwitch(self,switch_msg):
        self.active = switch_msg.data

    def cbImage(self,image_msg):
        if not self.active:
            return
        thread = threading.Thread(target=self.processImage,args=(image_msg,))
        thread.setDaemon(True)
        thread.start()

    def processImage(self, image_msg):
        if not self.thread_lock.acquire(False):
            return
        try:
            image_cv=self.bridge.imgmsg_to_cv2(image_msg,"bgr8")
        except CvBridgeErrer as e:
            print e
        img, detections = self.tm.contour_match(image_cv)
        detections.header.stamp = image_msg.header.stamp
        detections.header.frame_id = image_msg.header.frame_id
        self.pub_detections_list.publish(detections)
        height,width = img.shape[:2]
        try:
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)

        self.thread_lock.release()

if __name__=="__main__":
	rospy.init_node('badminton_detector_node')
	node = StaticObjectDetectorNode()
	rospy.spin()
