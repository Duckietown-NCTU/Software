#!/usr/bin/env python
from anti_instagram.AntiInstagram import *
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import (AntiInstagramTransform, BoolStamped, Segment,
    SegmentList, Vector2D, Twist2DStamped)
from duckietown_utils.instantiate_utils import instantiate
from duckietown_utils.jpg import image_cv_from_jpg
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import Marker
from line_detector.line_detector_plot import *
from line_detector.timekeeper import TimeKeeper
import cv2
import numpy as np
import rospy
import threading
import time


class ColorDetectorNode(object):
    def __init__(self):
        self.node_name = "ColorDetectorNode"

        # Thread lock 
        self.thread_lock = threading.Lock()
    
        # Constructor of line detector 
        self.bridge = CvBridge()

        self.active = True


        self.stats = Stats()

        # Only be verbose every 10 cycles
        self.intermittent_interval = 100
        self.intermittent_counter = 0

        # color correction
        self.ai = AntiInstagram()
 
        # these will be added if it becomes verbose
        self.pub_edge = None
        self.pub_colorSegment = None

        self.detector = None
        self.verbose = None
        self.updateParams(None)
   
         
        # Publishers
        self.pub_lane_recovery = rospy.Publisher("~lane_recovery", BoolStamped, queue_size=1)
        self.pub_image_green = rospy.Publisher("~green_hsv", Image, queue_size=1)
        self.pub_image_blue = rospy.Publisher("~blue_hsv", Image, queue_size=1)
        
        self.pub_car_cmd = rospy.Publisher("~car_cmd",Twist2DStamped,queue_size=1)
       
        # Subscribers
        self.sub_image = rospy.Subscriber("~image", CompressedImage, self.cbImage, queue_size=1)
        self.sub_transform = rospy.Subscriber("~transform", AntiInstagramTransform, self.cbTransform, queue_size=1)
        self.sub_switch = rospy.Subscriber("~switch", BoolStamped, self.cbSwitch, queue_size=1)

        rospy.loginfo("[%s] Initialized (verbose = %s)." %(self.node_name, self.verbose))

        rospy.Timer(rospy.Duration.from_sec(2.0), self.updateParams)


    def updateParams(self, _event):
        old_verbose = self.verbose
        self.verbose = rospy.get_param('~verbose', True)
        # self.loginfo('verbose = %r' % self.verbose)
        if self.verbose != old_verbose:
            self.loginfo('Verbose is now %r' % self.verbose)

        self.image_size = rospy.get_param('~img_size')
        self.top_cutoff = rospy.get_param('~top_cutoff')

        if self.detector is None:
            c = rospy.get_param('~detector')
            assert isinstance(c, list) and len(c) == 2, c
        
#         if str(self.detector_config) != str(c):
            self.loginfo('new detector config: %s' % str(c))

            self.detector = instantiate(c[0], c[1])
#             self.detector_config = c




    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data

    def cbImage(self, image_msg):
    	self.stop()

        self.stats.received()

        if not self.active:
            #print "******************** no color detector *********************"
            return 
        # Start a daemon thread to process the image
        thread = threading.Thread(target=self.processImage,args=(image_msg,))
        thread.setDaemon(True)
        thread.start()
        # Returns rightaway

    def cbTransform(self, transform_msg):
        self.ai.shift = transform_msg.s[0:3]
        self.ai.scale = transform_msg.s[3:6]

        self.loginfo("AntiInstagram transform received")

    def loginfo(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))

    def intermittent_log_now(self):
        return self.intermittent_counter % self.intermittent_interval == 1
    
    def intermittent_log(self, s):
        if not self.intermittent_log_now():
            return
        self.loginfo('%3d:%s' % (self.intermittent_counter, s))

    def processImage(self, image_msg):
        if not self.thread_lock.acquire(False):
            self.stats.skipped()
            # Return immediately if the thread is locked
            return

        try:
            self.processImage_(image_msg)
        finally:
            # Release the thread lock
            self.thread_lock.release()

    def processImage_(self, image_msg):

        self.stats.processed()

        if self.intermittent_log_now():
            self.intermittent_log(self.stats.info())
            self.stats.reset()

        tk = TimeKeeper(image_msg)
        
        self.intermittent_counter += 1

        # Decode from compressed image with OpenCV
        try:
            image_cv = image_cv_from_jpg(image_msg.data)
        except ValueError as e:
            self.loginfo('Could not decode image: %s' % e)
            return

        tk.completed('decoded')

        # Resize and crop image
        hei_original, wid_original = image_cv.shape[0:2]

        if self.image_size[0] != hei_original or self.image_size[1] != wid_original:
            # image_cv = cv2.GaussianBlur(image_cv, (5,5), 2)
            image_cv = cv2.resize(image_cv, (self.image_size[1], self.image_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
        image_cv = image_cv[self.top_cutoff:,:,:]

        tk.completed('resized')

        # apply color correction: AntiInstagram
        image_cv_corr = self.ai.applyTransform(image_cv)
        image_cv_corr = cv2.convertScaleAbs(image_cv_corr)

        tk.completed('corrected')

        # set up parameter

        hsv_green1 = np.array([70,100,60])
        hsv_green2 = np.array([90,255,255])
        hsv_blue1 = np.array([90,80,50])
        hsv_blue2 = np.array([110,255,255])


        # Set the image to be detected
        hsv = cv2.cvtColor(image_cv_corr,cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv,hsv_green1,hsv_green2)
        blue = cv2.inRange(hsv,hsv_blue1,hsv_blue2)
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        green = cv2.dilate(green, kernel)
        blue = cv2.dilate(blue, kernel)

        x = green[90:120,:]
        y = blue[90:120,:]
        
        msgg = BoolStamped()


        if (x==255).sum() > 250:
            print "green line detected!"
            time.sleep(4)
            print " 4 sec finish"

            msgg.data = True
            self.pub_lane_recovery.publish(msgg)


        elif (y==255).sum() > 250:
            print "blue line detected!"
            time.sleep(7)
            print " 7 sec finish"

            msgg.data = True
            self.pub_lane_recovery.publish(msgg)

        else:
            print "only red line detected"
            time.sleep(1)
            print " 1 sec finish"

            msgg.data = True
            self.pub_lane_recovery.publish(msgg)
       
        tk.completed('prepared')

        # VISUALIZATION only below
        
        if self.verbose:

                    

            tk.completed('drawn')

            # Publish the frame with lines

            image_msg_out_green = self.bridge.cv2_to_imgmsg(green, "mono8")
            image_msg_out_green.header.stamp = image_msg.header.stamp
            self.pub_image_green.publish(image_msg_out_green)

            image_msg_out_blue = self.bridge.cv2_to_imgmsg(blue, "mono8")
            image_msg_out_blue.header.stamp = image_msg.header.stamp
            self.pub_image_blue.publish(image_msg_out_blue)

            tk.completed('pub_image')




        self.intermittent_log(tk.getall())


    def onShutdown(self):
        self.loginfo("Shutdown.")
            
    def stop(self):
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0.0
        car_control_msg.omega = 0.0
        self.pub_car_cmd.publish(car_control_msg)


class Stats():
    def __init__(self):
        self.nresets = 0
        self.reset()

    def reset(self):
        self.nresets += 1
        self.t0 = time.time()
        self.nreceived = 0
        self.nskipped = 0
        self.nprocessed = 0

    def received(self):
        if self.nreceived == 0 and self.nresets == 1:
            rospy.loginfo('line_detector_node received first image.')
        self.nreceived += 1

    def skipped(self):
        self.nskipped += 1

    def processed(self):
        if self.nprocessed == 0 and self.nresets == 1:
            rospy.loginfo('line_detector_node processing first image.')

        self.nprocessed += 1

    def info(self):
        delta = time.time() - self.t0

        if self.nreceived:
            skipped_perc = (100.0 * self.nskipped / self.nreceived)
        else:
            skipped_perc = 0

        def fps(x):
            return '%.1f fps' % (x / delta)

        m = ('In the last %.1f s: received %d (%s) processed %d (%s) skipped %d (%s) (%1.f%%)' %
             (delta, self.nreceived, fps(self.nreceived),
              self.nprocessed, fps(self.nprocessed),
              self.nskipped, fps(self.nskipped), skipped_perc))
        return m





if __name__ == '__main__': 
    rospy.init_node('color_detector',anonymous=False)
    color_detector_node = ColorDetectorNode()
    rospy.on_shutdown(color_detector_node.onShutdown)
    rospy.spin()



