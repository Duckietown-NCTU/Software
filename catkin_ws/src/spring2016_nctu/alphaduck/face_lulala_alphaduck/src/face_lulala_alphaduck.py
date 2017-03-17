#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Segment, SegmentList, Vector2D
from line_detector.WhiteBalance import *
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import threading
#from PIL import Image as pimg
#import jpeg4py as jpeg

class FaceLulalaAlphaduckNode(object):
    def __init__(self):
        self.node_name = "face_lulala_alphaduck"

        # Thread lock 
        self.thread_lock = threading.Lock()
       
        # Constructor of line detector 
        self.bridge = CvBridge()
#       self.detector = LineDetector()
        self.wb = WhiteBalance()
        self.flag_wb_ref = False
       
        # Parameters
#       self.flag_wb = rospy.get_param('~white_balance')
        
#       self.image_size = rospy.get_param('~img_size')
#       self.top_cutoff = rospy.get_param('~top_cutoff')
""" 
        self.detector.hsv_white1 = np.array(rospy.get_param('~hsv_white1'))
        self.detector.hsv_white2 = np.array(rospy.get_param('~hsv_white2'))
        self.detector.hsv_yellow1 = np.array(rospy.get_param('~hsv_yellow1'))
        self.detector.hsv_yellow2 = np.array(rospy.get_param('~hsv_yellow2'))
        self.detector.hsv_red1 = np.array(rospy.get_param('~hsv_red1'))
        self.detector.hsv_red2 = np.array(rospy.get_param('~hsv_red2'))
        self.detector.hsv_red3 = np.array(rospy.get_param('~hsv_red3'))
        self.detector.hsv_red4 = np.array(rospy.get_param('~hsv_red4'))

        self.detector.dilation_kernel_size = rospy.get_param('~dilation_kernel_size')
        self.detector.canny_thresholds = rospy.get_param('~canny_thresholds')
        self.detector.hough_min_line_length = rospy.get_param('~hough_min_line_length')
        self.detector.hough_max_line_gap    = rospy.get_param('~hough_max_line_gap')
        self.detector.hough_threshold = rospy.get_param('~hough_threshold')
"""
        # Publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
       
        # Verbose option 
        self.verbose = rospy.get_param('~verbose')
        if self.verbose:
            self.toc_pre = rospy.get_time()   

        # Subscribers
        self.sub_image = rospy.Subscriber("~image", CompressedImage, self.cbImage, queue_size=1)
        rospy.loginfo("[%s] Initialized." %(self.node_name))

    def cbImage(self,image_msg):
        # Start a daemon thread to process the image
        thread = threading.Thread(target=self.processImage,args=(image_msg,))
        thread.setDaemon(True)
        thread.start()
        # Returns rightaway

    def processImage(self,image_msg):
        if not self.thread_lock.acquire(False):
            # Return immediately if the thread is locked
            return
        
        # Verbose
        if self.verbose:
            rospy.loginfo("[%s] Latency received = %.3f ms" %(self.node_name, (rospy.get_time()-image_msg.header.stamp.to_sec()) * 1000.0))
        
        # time_start = rospy.Time.now()
        # time_start = event.last_real
        # msg_age = time_start - image_msg.header.stamp
        # rospy.loginfo("[LineDetector] image age: %s" %msg_age.to_sec())

        # Decode from compressed image
        # with OpenCV
        image_cv = cv2.imdecode(np.fromstring(image_msg.data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        
        # with PIL Image
        # image_cv = jpeg.JPEG(np.fromstring(image_msg.data, np.uint8)).decode()
        
        # with libjpeg-turbo
        # Convert from uncompressed image message
        # image_cv = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        
        # Verbose
        if self.verbose:
            self.tic = rospy.get_time()   
            rospy.loginfo("[%s] Latency image decompressed = %.3f ms" %(self.node_name, (self.tic-image_msg.header.stamp.to_sec()) * 1000.0))
        
        # White balancing: set reference image to estimate parameters
        if self.flag_wb and (not self.flag_wb_ref):
            # set reference image to estimate parameters
            self.wb.setRefImg(image_cv)
            rospy.loginfo("[%s] White balance: parameters computed." %(self.node_name))
            print self.wb.norm_bgr
            self.flag_wb_ref = True

        # Resize and crop image
        hei_original = image_cv.shape[0]
        wid_original = image_cv.shape[1]
        if self.image_size[0]!=hei_original or self.image_size[1]!=wid_original:
            # image_cv = cv2.GaussianBlur(image_cv, (5,5), 2)
            image_cv = cv2.resize(image_cv, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        image_cv = image_cv[self.top_cutoff:,:,:]

        # White balancing
        if self.flag_wb and self.flag_wb_ref:
            self.wb.correctImg(image_cv)

        # Set the image to be detected
        self.detector.setImage(image_cv)
	
        # Detect lines and normals
#       lines_white, normals_white = self.detector.detectLines('white')
#       lines_yellow, normals_yellow = self.detector.detectLines('yellow')
#       lines_red, normals_red = self.detector.detectLines('red')

        # Draw lines and normals
#       self.detector.drawLines(lines_white, (0,0,0))
#       self.detector.drawLines(lines_yellow, (255,0,0))
#       self.detector.drawLines(lines_red, (0,255,0))
        #self.detector.drawNormals(lines_white, normals_white)
        #self.detector.drawNormals(lines_yellow, normals_yellow)
        #self.detector.drawNormals(lines_red, normals_red)

        # SegmentList constructor
#       segmentList = SegmentList()
#       segmentList.header.stamp = image_msg.header.stamp
        
        # Convert to normalized pixel coordinates, and add segments to segmentList
#       arr_cutoff = np.array((0, self.top_cutoff, 0, self.top_cutoff))
#       arr_ratio = np.array((1./self.image_size[1], 1./self.image_size[0], 1./self.image_size[1], 1./self.image_size[0]))
#       if len(lines_white)>0:
#           lines_normalized_white = ((lines_white + arr_cutoff) * arr_ratio)
#           segmentList.segments.extend(self.toSegmentMsg(lines_normalized_white, normals_white, Segment.WHITE))
#       if len(lines_yellow)>0:
#           lines_normalized_yellow = ((lines_yellow + arr_cutoff) * arr_ratio)
#           segmentList.segments.extend(self.toSegmentMsg(lines_normalized_yellow, normals_yellow, Segment.YELLOW))
#       if len(lines_red)>0:
#           lines_normalized_red = ((lines_red + arr_cutoff) * arr_ratio)
#           segmentList.segments.extend(self.toSegmentMsg(lines_normalized_red, normals_red, Segment.RED))
        
        # Verbose
        if self.verbose:
            self.toc = rospy.get_time()
            rospy.loginfo("[%s] Image processing time: %.3f ms" %(self.node_name, (self.toc-self.tic)*1000.0))
            rospy.loginfo("[%s] Number of white segments = %d" %(self.node_name, len(lines_white)))
            rospy.loginfo("[%s] number of yellow segments = %d" %(self.node_name, len(lines_yellow)))
            rospy.loginfo("[%s] number of red segments = %d" %(self.node_name, len(lines_red)))
            self.toc_pre = self.toc
 
        # Publish segmentList
        self.pub_lines.publish(segmentList)
        # time_spent = rospy.Time.now() - time_start
        # rospy.loginfo("[LineDetectorNode] Spent: %s" %(time_spent.to_sec()))
         
        # Publish the frame with lines
        image_msg_out = self.bridge.cv2_to_imgmsg(self.detector.getImage(), "bgr8")
        image_msg_out.header.stamp = image_msg.header.stamp
        self.pub_image.publish(image_msg_out)
        # time_spent = rospy.Time.now() - time_start
        # rospy.loginfo("[LineDetectorNode] Spent on img: %s" %(time_spent.to_sec()))

        # Verbose
        if self.verbose:
            rospy.loginfo("[%s] Latency sent = %.3f ms" %(self.node_name, (rospy.get_time()-image_msg.header.stamp.to_sec()) * 1000.0))

        # Release the thread lock
        self.thread_lock.release()

    def onShutdown(self):
            rospy.loginfo("[LineDetectorNode] Shutdown.")
            
    def toSegmentMsg(self,  lines, normals, color):
        
        segmentMsgList = []
        for x1,y1,x2,y2,norm_x,norm_y in np.hstack((lines,normals)):
            segment = Segment()
            segment.color = color
            segment.pixels_normalized[0].x = x1
            segment.pixels_normalized[0].y = y1
            segment.pixels_normalized[1].x = x2
            segment.pixels_normalized[1].y = y2
            segment.normal.x = norm_x
            segment.normal.y = norm_y
             
            segmentMsgList.append(segment)
        return segmentMsgList
 
if __name__ == '__main__': 
    rospy.init_node('face_lulala_alphaduck',anonymous=False)
    face_lulala_alphaduck_node = FaceLulalaAlphaduckNode()
    rospy.on_shutdown(face_lulala_alphaduck_node.onShutdown)
    rospy.spin()
