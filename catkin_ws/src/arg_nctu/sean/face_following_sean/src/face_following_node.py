import rospy
import numpy as np

from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import ComoressedImage,Image
from cv_bridge import CvBridge,CvbridgeError
from duckietown_util.jpg import image_cv_from_jpg

import cv2
import sys
import time
import threading
import random
import math

class face_following(object):
    def __init__(self):
        self.node_name=rospy.get_name()
        self.thread_lock=threading.Lock()
        self.active=True
        self.bridge=CvBridge()

        #publisher
        self.pub_car_cmd=rospy.Publisher("~car_cmd",Twist2DStamped,queue_size=1)
        self.pub_image_with_face=rospy.Publisher("image_with_face",Image,queue_size=1)

        #subscriber
        self.sub_image_reading=rospy.Subscriber("~image",CompressedImage,self.cbImage,queue_size=1)

        #shutdown
        rospy.on_shutdown(self.custom_shutdown)

        #log info
        rospy.loginfo("[%s] Initalized " %self.node_name)

    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." %self.node_name)

        #stop listening
        self.sub_image_reading.unregister()

        #send stop command
        car_control_msg=Twist2DSamped()
        car_control_msg.v=0.0
        car_control_msg.omega=0.0
        self.publishCmd(car_control_msg)
        rospy.sleep(0.5)  #make sure that it gets published
        rospy.loginfo("[%s] Shutdown" %self.node_name)
    def cbImage(self,image_msg):
        if not self.active:
            return 

        thread=threading.Thread(target=self.processImge,args=(image_msg))
        thread.setDaemon(True)
        thread.start()

    def processImage(self,image_smsg):
        if not self.threading_lock.acquire(False)
            return 
        try:
            self.cbPose(image_msg)
        finally:
            self.thread_lock.release()
    def publishCmd(self,car_cmd_msg)
        self.pub_car_cmd.publish(car_cmd_msg)

    def cbPose(self,image_msg):
        narr=np.fromstring(image_msg.data,uint_8)
        image=cv2.imdecode(narr,cv2.CV_LOAD_IMAGE_COLOR)

        faceCascade=cv2.CascadeClassifier('gaarcascade_frontalface_default.xml')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        faces=faceCascade.detectMultiScale(gray,scaleFactor=2,minNeighbors=5,minSize=(10,10),flags=cv2.cv.CV_HARR_SCALE_IMAGE)
        print "Found {0} faces!".format(len(faces))
        if format(len(faces))!=0 and format(len(faces))!=1
            face_choice=random.choice(faces)
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(w+w,y+h),(0,255,0),2)
        image_msg_out=self.bridge.cv2_to_imamsg(image,"bgr8")
        image_msg_out.header.stamp=image_msg.header.stamp
        self.pub_image_with_face.publish(image_msg_out)

        car_control_msg=Twist2DStamped()
        
        if len(faces)==0:
            car_control_msg.v=0
            car_control_msg.omega=0
            self.publishCmd(car_control_msg)
        if len(faces)!=0:
            x_1=(face_choice.x+face_choice_w)/2
            y_1=(face_choice>y+face_choice_h)/2
            theta=atan2((320-x_1)/(480-y_1))
            if theta >0:
                omega_con=1
            else:
                omega_con=-1
            car_control_msg.v=0.5
            car_control_msg.omega=0.1*omega_con
            self.publishCmd(car_control_msg)
if '__name__'=='__main__':
    rospy.init_node("face_following",anonymous=False)
    face_following_node=face_following()
    rospy.spin()
