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
import os
import DetectChars
import DetectPlates
import PossiblePlate

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
        self.pub_image_original = rospy.Publisher("~image_with_plate", Image, queue_size=1)
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
        
        narr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(narr, cv2.CV_LOAD_IMAGE_COLOR)

##############################################################################################
	# module level variables ##########################################################################
	SCALAR_BLACK = (0.0, 0.0, 0.0)
	SCALAR_WHITE = (255.0, 255.0, 255.0)
	SCALAR_YELLOW = (0.0, 255.0, 255.0)
	SCALAR_GREEN = (0.0, 255.0, 0.0)
	SCALAR_RED = (0.0, 0.0, 255.0)

	showSteps = False	
	
	blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

	if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
		print "\nerror: KNN traning was not successful\n"               # show error message
		return                                                          # and exit program
	    # end if

	imgOriginalScene  = image  #cv2.imread("1.png")               # open image

	#if imgOriginalScene is None:                            # if image was not read successfully
		#print "\nerror: image not read from file \n\n"      # print error message to std out
		#os.system("pause")                                  # pause so user can see error message
		#return                                              # and exit program
		# end if

	listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

	listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

	cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

	if len(listOfPossiblePlates) == 0:                          # if no plates were found
		print "\nno license plates were detected\n"             # inform user no plates were found
	else:                                                       # else
		# if we get in here list of possible plates has at leat one plate
		# sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
		listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

		# suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
		licPlate = listOfPossiblePlates[0]

		cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
		cv2.imshow("imgThresh", licPlate.imgThresh)

		if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
		    print "\nno characters were detected\n\n"       # show message
		    return                                          # and exit program
		# end if

		drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

		print "\nlicense plate read from image = " + licPlate.strChars + "\n"       # write license plate text to std out
		print "----------------------------------------"

		writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

		cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

		#cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file
		image = imgOriginalScene

	    # end if else

##############################################################################################

#        print "Found {0} faces!".format(len(faces))
        
#        for (x, y, w, h) in faces:
#           cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #   cv2.imshow("preview", image)
        #   cv2.waitKey(0)
        #   cv2.destroyAllWindows()
        

	#cv2.rectangle(image, (100, 100), (120, 140), (0, 255, 0), 2)
        #image_msg_out = self.bridge.cv2_to_imgmsg(mask, "mono8")
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


        # debuging
        # self.pub_counter += 1
        # if self.pub_counter % 50 == 0:
        #     self.pub_counter = 1
        #     print "lane_controller publish"
        #     print car_control_msg

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function
###################################################################################################
###################################################################################################
if __name__ == "__main__":
    rospy.init_node("lane_controller",anonymous=False)
    lane_control_node = lane_controller()
    rospy.spin()
