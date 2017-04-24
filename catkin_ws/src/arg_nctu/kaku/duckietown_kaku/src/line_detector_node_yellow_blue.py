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


class LineDetectorNode(object):
    def __init__(self):
        self.node_name = "LineDetectorNode"

        # Thread lock 
        self.thread_lock = threading.Lock()
    
        # Constructor of line detector 
        self.bridge = CvBridge()

        self.active = True
        self.time_switch = True

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
   
        self.blue = 0
        self.yellow = 0
        self.count = 0
        
 
        # Publishers
        self.pub_lines = rospy.Publisher("~segment_list", SegmentList, queue_size=1)
        self.pub_image = rospy.Publisher("~image_with_lines", Image, queue_size=1)
        self.pub_lane = rospy.Publisher("~have_lines", BoolStamped, queue_size=1, latch=True)
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

        if self.verbose and self.pub_edge is None:
            self.pub_edge = rospy.Publisher("~edge", Image, queue_size=1)
            self.pub_colorSegment = rospy.Publisher("~colorSegment", Image, queue_size=1)


    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data

    def cbImage(self, image_msg):
        self.stats.received()

        if not self.active:
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

        hsv_blue1 = (100,50,50)
        hsv_blue2 = (150,255,255)
        hsv_yellow1 = (25,50,50)
        hsv_yellow2 = (45,255,255)


        # Set the image to be detected
        gray = cv2.cvtColor(image_cv_corr,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200, apertureSize = 3)
    
        hsv = cv2.cvtColor(image_cv_corr, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, hsv_yellow1, hsv_yellow2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        yellow = cv2.dilate(yellow, kernel)

        blue = cv2.inRange(hsv, hsv_blue1, hsv_blue2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        blue = cv2.dilate(blue, kernel)

        # Detect lines and normals

        edge_color_yellow = cv2.bitwise_and(yellow, edges)
        lines_yellow = cv2.HoughLinesP(edge_color_yellow, 1, np.pi/180, 10, np.empty(1), 3, 1)
        if lines_yellow is not None:
            lines_yellow = np.array(lines_yellow[0])
        
        else:
            lines_yellow = []

        #edge_color_blue = cv2.bitwise_and(blue, edges)
        #lines_blue = cv2.HoughLinesP(edge_color_blue, 1, np.pi/180, 10, np.empty(1), 3, 1)
        #if lines_blue is not None:
            #lines_blue = np.array(lines_blue[0])

        #else:
            #lines_blue = []

	#print "***************** ",image_cv.shape," *********************"
        #bw_yellow = yellow
        #bw_blue = blue

        self.blue = blue
        self.yellow = yellow
        if len(lines_yellow) > 0:
            lines_yellow,normals_yellow = self.normals(lines_yellow,yellow)

        #if len(lines_blue) > 0:
            #lines_blue,normals_blue = self.normals(lines_blue,bw_blue)


        tk.completed('detected')
     
        # SegmentList constructor
        segmentList = SegmentList()
        segmentList.header.stamp = image_msg.header.stamp
        
        # Convert to normalized pixel coordinates, and add segments to segmentList
        
        
        if len(lines_yellow) > 0:
            segmentList.segments.extend(self.toSegmentMsg(lines_yellow, normals_yellow, Segment.YELLOW))
        if len(segmentList.segments) == 0:

                    if self.time_switch == False:
                        msgg = BoolStamped()
                        msgg.data = False
                        self.pub_lane.publish(msgg)
                        self.time_switch = True


                    car_control_msg = Twist2DStamped()
                    car_control_msg.v = 0.0
                    car_control_msg.omega = 0.0
                    self.pub_car_cmd.publish(car_control_msg)
                

        #if len(lines_blue) > 0:
            #segmentList.segments.extend(self.toSegmentMsg(lines_blue, normals_blue, Segment.YELLOW))


        self.intermittent_log('# segments:yellow %3d' % (len(lines_yellow)))

        tk.completed('prepared')

        # Publish segmentList
        self.pub_lines.publish(segmentList)

        # VISUALIZATION only below
        
        if self.verbose:

            # Draw lines and normals
            image_with_lines = np.copy(image_cv_corr)
            for x1,y1,x2,y2,norm_x,norm_y in np.hstack((lines_yellow,normals_yellow)):
                x1 = int(x1) 
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
        
                ox= int((x1+x2)/2)
                oy= int((y1+y2)/2)
        
                cx = (ox+3*norm_x).astype('int')
                cy = (oy+3*norm_y).astype('int') 
        
                ccx = (ox-3*norm_x).astype('int')
                ccy = (oy-3*norm_y).astype('int') 
        
                if cx >158:
                    cx = 158
                elif cx <1:
                    cx = 1
                if ccx >158:
                    ccx = 158
                elif ccx <1:
                    ccx = 1
 
                if cy >=79:
                    cy = 79
                elif cy <1:
                    cy = 1
                if ccy >=79:
                    ccy = 79
                elif ccy <1:
                    ccy = 1

        
                if (blue[cy, cx] == 255 and yellow[ccy,ccx] ==255) or (yellow[cy, cx] == 255 and blue[ccy,ccx] ==255):

                    cv2.line(image_with_lines, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.circle(image_with_lines, (x1,y1), 2, (0,255,0))
                    cv2.circle(image_with_lines, (x2,y2), 2, (255,0,0))            

            tk.completed('drawn')

            # Publish the frame with lines
            image_msg_out = self.bridge.cv2_to_imgmsg(image_with_lines, "bgr8")
            image_msg_out.header.stamp = image_msg.header.stamp
            self.pub_image.publish(image_msg_out)

            tk.completed('pub_image')

#         if self.verbose:
            #colorSegment = color_segment(white.area, red.area, yellow.area) 
            #edge_msg_out = self.bridge.cv2_to_imgmsg(self.detector.edges, "mono8")
            #colorSegment_msg_out = self.bridge.cv2_to_imgmsg(colorSegment, "bgr8")
            #self.pub_edge.publish(edge_msg_out)
            #self.pub_colorSegment.publish(colorSegment_msg_out)

            tk.completed('pub_edge/pub_segment')


        self.intermittent_log(tk.getall())


    def onShutdown(self):
        self.loginfo("Shutdown.")
            
    def toSegmentMsg(self,  lines, normals, color):

    	arr_cutoff = np.array((0, self.top_cutoff, 0, self.top_cutoff))
        arr_ratio = np.array((1./self.image_size[1], 1./self.image_size[0], 1./self.image_size[1], 1./self.image_size[0]))
        
        segmentMsgList = []
        for x1,y1,x2,y2,norm_x,norm_y in np.hstack((lines,normals)):

            ox= int((x1+x2)/2)
            oy= int((y1+y2)/2)
        
            cx = (ox+3*norm_x).astype('int')
            cy = (oy+3*norm_y).astype('int') 
        
            ccx = (ox-3*norm_x).astype('int')
            ccy = (oy-3*norm_y).astype('int') 
        
            if cx >158:
                cx = 158
            elif cx <1:
                cx = 1
            if ccx >158:
                ccx = 158
            elif ccx <1:
                ccx = 1

            if cy >=79:
                cy = 79
            elif cy <1:
                cy = 1
            if ccy >=79:
                ccy = 79
            elif ccy <1:
                ccy = 1

        
            if (self.blue[cy, cx] == 255 and self.yellow[ccy,ccx] ==255) or (self.yellow[cy, cx] == 255 and self.blue[ccy,ccx] ==255):

                [x1,y1,x2,y2] = (([x1,y1,x2,y2] + arr_cutoff) * arr_ratio)

                segment = Segment()
                segment.color = color
                segment.pixels_normalized[0].x = x1
                segment.pixels_normalized[0].y = y1
                segment.pixels_normalized[1].x = x2
                segment.pixels_normalized[1].y = y2
                segment.normal.x = norm_x
                segment.normal.y = norm_y
                segmentMsgList.append(segment)

                if self.time_switch == True:
                    msgg = BoolStamped()
                    msgg.data = True
                    self.pub_lane.publish(msgg)
                    self.time_switch = False
                    self.count = 0

            

        return segmentMsgList

    def normals(self, lines,bw): 

        if len(lines) >0:

            normals = []

            centers = []

            #find the dx dy

            length = np.sum((lines[:, 0:2] -lines[:, 2:4])**2, axis=1, keepdims=True)**0.5

            dx = 1.* (lines[:,3:4]-lines[:,1:2])/length

            dy = 1.* (lines[:,0:1]-lines[:,2:3])/length

            centers = np.hstack([(lines[:,0:1]+lines[:,2:3])/2, (lines[:,1:2]+lines[:,3:4])/2])


            x3 = (centers[:,0:1] - 3.*dx).astype('int')

            x3[x3<0]=0

            x3[x3>=160]=160-1

            y3 = (centers[:,1:2] - 3.*dy).astype('int')
    
            y3[y3<0]=0

            y3[y3>=80]=80-1

            x4 = (centers[:,0:1] + 3.*dx).astype('int')
        
            x4[x4<0]=0
    
            x4[x4>=160]=160-1

            y4 = (centers[:,1:2] + 3.*dy).astype('int')

            y4[y4<0]=0

            y4[y4>=80]=80-1

            flag_signs = (np.logical_and(bw[y3,x3]>0, bw[y4,x4]==0)).astype('int')*2-1

            normals = np.hstack([dx, dy]) * flag_signs  

            flag = ((lines[:,2]-lines[:,0])*normals[:,1] - (lines[:,3]-lines[:,1])*normals[:,0])>0

            for i in range(len(lines)):

                if flag[i]:

                    x1,y1,x2,y2 = lines[i, :]

                    lines[i, :] = [x2,y2,x1,y1]

            return lines,normals


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
    rospy.init_node('line_detector',anonymous=False)
    line_detector_node = LineDetectorNode()
    rospy.on_shutdown(line_detector_node.onShutdown)
    rospy.spin()



