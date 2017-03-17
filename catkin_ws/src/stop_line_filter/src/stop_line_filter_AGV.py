#!/usr/bin/env python
import rospy
import numpy as np
from duckietown_msgs.msg import SegmentList, Segment, BoolStamped, StopLineReading, LanePose, FSMState
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
import time
import math

class StopLineFilterNode(object):
    def __init__(self):
        self.node_name = "Stop Line Filter"
        self.active = True
        ## state vars
        self.lane_pose = LanePose()

        ## params
        self.stop_distance = self.setupParam("~stop_distance", 0.2) # distance from the stop line that we should stop 
        self.min_segs      = self.setupParam("~min_segs", 2) # minimum number of red segments that we should detect to estimate a stop
        self.off_time      = self.setupParam("~off_time", 2)
        self.lanewidth = 0 # updated continuously below

        self.state = "JOYSTICK_CONTROL"
        self.sleep = False
        ## publishers and subscribers
        self.sub_segs      = rospy.Subscriber("~segment_list", SegmentList, self.processSegments)
        self.sub_lane      = rospy.Subscriber("~lane_pose",LanePose, self.processLanePose)
        self.sub_mode      = rospy.Subscriber("fsm_node/mode",FSMState, self.processStateChange)
        self.pub_stop_line_reading = rospy.Publisher("~stop_line_reading", StopLineReading, queue_size=1)
        self.pub_at_stop_line = rospy.Publisher("~at_stop_line", BoolStamped, queue_size=1)
	self.sub_switch = rospy.Subscriber("~switch", BoolStamped, self.cbSwitch, queue_size=1)


        self.params_update = rospy.Timer(rospy.Duration.from_sec(1.0), self.updateParams)

    def setupParam(self,param_name,default_value):
        value = rospy.get_param(param_name,default_value)
        rospy.set_param(param_name,value) #Write to parameter server for transparancy
        rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value

    def updateParams(self,event):
        self.lanewidth = rospy.get_param("~lanewidth")
        self.stop_distance = rospy.get_param("~stop_distance")
        self.min_segs      = rospy.get_param("~min_segs")
        self.off_time      = rospy.get_param("~off_time")

    def processStateChange(self, msg):
        if self.state == "INTERSECTION_CONTROL" and (msg.state == "LANE_FOLLOWING" or msg.state == "PARALLEL_AUTONOMY"):
            rospy.loginfo("stop line sleep start")
            self.sleep = True
            rospy.sleep(self.off_time)
            self.sleep = False
            rospy.loginfo("stop line sleep end")
        self.state=msg.state

    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data

    def processLanePose(self, lane_pose_msg):
        self.lane_pose = lane_pose_msg

    def processSegments(self, segment_list_msg):
        good_seg_count=0
        stop_line_x_accumulator=0.0
        stop_line_y_accumulator=0.0
        for segment in segment_list_msg.segments:
            if segment.color != segment.RED:
                continue
            if segment.points[0].x < 0 or segment.points[1].x < 0: # the point is behind us 
                continue
	
 	    if not self.active or self.sleep:
            	continue

            p1_lane = self.to_lane_frame(segment.points[0])
            p2_lane = self.to_lane_frame(segment.points[1])
            avg_x = 0.5*(p1_lane[0] + p2_lane[0])
            avg_y = 0.5*(p1_lane[1] + p2_lane[1])
            stop_line_x_accumulator += avg_x
            stop_line_y_accumulator += avg_y # TODO output covariance and not just mean
            good_seg_count += 1.0

        stop_line_reading_msg = StopLineReading()
        stop_line_reading_msg.header.stamp = segment_list_msg.header.stamp
        if (good_seg_count < self.min_segs):
            stop_line_reading_msg.stop_line_detected = False
            stop_line_reading_msg.at_stop_line = False
            self.pub_stop_line_reading.publish(stop_line_reading_msg)
            return
        
        stop_line_reading_msg.stop_line_detected = True
        stop_line_point = Point()
        stop_line_point.x = stop_line_x_accumulator/good_seg_count
        stop_line_point.y = stop_line_y_accumulator/good_seg_count
        stop_line_reading_msg.stop_line_point = stop_line_point
        stop_line_reading_msg.at_stop_line = stop_line_point.x < self.stop_distance and math.fabs(stop_line_point.y) < self.lanewidth/4 
        self.pub_stop_line_reading.publish(stop_line_reading_msg)    
        if stop_line_reading_msg.at_stop_line:
            msg = BoolStamped()
            msg.header.stamp = stop_line_reading_msg.header.stamp
            msg.data = True
            self.pub_at_stop_line.publish(msg)
   
    def to_lane_frame(self, point):
        p_homo = np.array([point.x,point.y,1])
        phi = self.lane_pose.phi
        d   = self.lane_pose.d
        T = np.array([[math.cos(phi), -math.sin(phi), 0],
                      [math.sin(phi), math.cos(phi) , d],
                      [0,0,1]])
        p_new_homo = T.dot(p_homo)
        p_new = p_new_homo[0:2]
        return p_new
 
    def onShutdown(self):
        rospy.loginfo("[StopLineFilterNode] Shutdown.")

if __name__ == '__main__': 
    rospy.init_node('stop_line_filter',anonymous=False)
    lane_filter_node = StopLineFilterNode()
    rospy.on_shutdown(lane_filter_node.onShutdown)
    rospy.spin()

