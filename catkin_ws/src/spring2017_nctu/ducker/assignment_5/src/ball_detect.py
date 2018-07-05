#!/usr/bin/env python
import rospy
import signal
import sys
import cv2
import cv2.cv as cv
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import numpy as np

TAG = "Ball Detection:"

#-------------------------- YELLOW BALL DETECTOR ------------------------------#
def joy_callback(data):
    global pub
    print data.axes[1], data.axes[4]

    msg = WheelsCmdStamped()
    msg.vel_left = float(data.axes[1])
    msg.vel_right = float(data.axes[4])

    h = Header()
    h.stamp = rospy.Time.now()
    msg.header = h

    pub.publish(msg)

def image_handler(data):
    global pub
    print "IMAGE"
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")

    scaled_image = cv2.resize(cv_image, None,fx=0.5, fy=0.5)
    gray_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2HSV)

    mask =  cv2.inRange(hsv_image, cv.Scalar(10, 100, 100), cv.Scalar(70, 255, 255))
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.bitwise_not(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(gray_image,contours,-1,(0,255,0),3)

    for cntr in contours:
        (x,y),radius = cv2.minEnclosingCircle(cntr)
        approx = cv2.approxPolyDP(cntr,0.01*cv2.arcLength(cntr,True),True)
        if len(approx)>10:
            print "L:",len(approx)
            print "FALL FOUND:","X:",int(x),"Y",int(y),"R:",int(radius)
            cv2.circle(gray_image,(int(x),int(y)),int(radius),(0,0,255),3)

    image_message = bridge.cv2_to_imgmsg(gray_image, encoding="8UC1")
    pub.publish(image_message)

# SIGINT Signal handler
def sigint_handler(signal, frame):
        global pub
        print ""
        print TAG,"Interrupt!"
        print TAG,"Terminated"
        sys.exit(0)

if __name__ == '__main__':
    print TAG,"Started"
    params = cv2.SimpleBlobDetector_Params()
    print params.filterByColor
    print params.filterByArea
    print params.filterByCircularity
    print params.filterByInertia
    print params.filterByConvexity
    # Assigning the SIGINT handler.
    signal.signal(signal.SIGINT, sigint_handler)
    # Starting the node
    rospy.init_node('ball_detector', anonymous=True)
    rospy.Subscriber("/ducktruck/camera_node/image_rect_color",Image,image_handler)
    pub = rospy.Publisher("/result_image",Image,queue_size=5)
    rospy.spin()

    print TAG,"Terminated"
