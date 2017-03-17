#!/usr/bin/env python
import rospy
import numpy as np
from duckietown_msgs.msg import BoolStamped, FSMState, Twist2DStamped
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
import math
import time

class Project(object):
    def __init__(self):
        self.node_name = "Project"
        self.active = True

        self.stack = ['B']
        
        ## publishers and subscribers
        self.sub_mode = rospy.Subscriber("/buyme/fsm_node/mode",FSMState, self.processStateChange)
        self.sub_in_lane = rospy.Subscriber("/buyme/lane_filter_node/in_lane", BoolStamped, self.process)
        self.sub_switch = rospy.Subscriber("/buyme/project/switch", BoolStamped, self.cbSwitch, queue_size=1)
        
        self.pub_last_is_forward = rospy.Publisher("~last_is_forward", BoolStamped, queue_size=1, latch=True)
        self.pub_last_is_turn = rospy.Publisher("~last_is_turn", BoolStamped, queue_size=1, latch=True)
        self.pub_time_is_up = rospy.Publisher("~time_is_up", BoolStamped, queue_size=1, latch=True)
        self.pub_car_cmd = rospy.Publisher("/buyme/motion_planning/car_cmd",Twist2DStamped,queue_size=1)

    def processStateChange(self, msg):
        self.state=msg.state

    def car_command(self, v, omega, duration):
        # Send stop command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v 
        car_control_msg.omega = omega 
        self.pub_car_cmd.publish(car_control_msg)
        rospy.sleep(duration)
        #rospy.loginfo("Shutdown")
        car_control_msg.v = 0.0 
        car_control_msg.omega = 0.0 
        self.pub_car_cmd.publish(car_control_msg)

    class switch(object):
        def __init__(self, value):
            self.value = value
            self.fall = False
                                 
        def __iter__(self):
            yield self.match
            raise StopIteration
                                                                  
        def match(self, *args):
            if self.fall or not args:
                return True
            elif self.value in args: 
                self.fall = True
                return True
            else:
                return False

    def motion_planning(self, concat):
        for i in range(len(concat)):
            primitives = concat[i]
            for case in self.switch(primitives):
                if case('F'):
                    self.car_command(0.6, 0, 0.78)
                    break
                if case('L'):
                    self.car_command(0.2, 4, 0.82)
                    break
                if case('R'):
                    self.car_command(0, -3.5, 1.0)
                    break
                if case('B'):
                    self.car_command(-0.6, 0, 0.78)
                    break

    def process(self, msgg):
        if self.active == True:
            print "##################fsm state is " + `self.state`
     
            if self.state == "RESET":
                self.stack = ['B']
                time.sleep(2)
            elif self.state == "FORWARD":
                self.motion_planning("F")
                self.stack.append('F')
                time.sleep(2)	    
            elif self.state == "TURN_RIGHT":
                if self.stack.pop() == 'R':
                    self.stack.append('R')
                    msg = BoolStamped()
                    msg.data = True
                    self.pub_time_is_up.publish(msg)
                else:
                    self.motion_planning("R")
                    self.stack.append('R')
                time.sleep(2)	    
            elif self.state == "TURN_BACK":
                if self.stack.pop() == 'L':
                    self.stack.append('L')
                    msg = BoolStamped()
                    msg.data = True
                    self.pub_time_is_up.publish(msg)
                else:
                    self.motion_planning("RR")
                    self.stack.append('L')
                time.sleep(2)	    
            elif self.state == "TRACE_BACK":
                self.motion_planning("RB")
                self.stack.pop()
                c = self.stack.pop()
                if c == 'F':
                    self.stack.append('F')
                    msg = BoolStamped()
                    msg.data = True
                    self.pub_last_is_forward.publish(msg)
                    msg.data = False
                    self.pub_last_is_turn.publish(msg)
                elif c == 'R':
                    msg = BoolStamped()
                    msg.data = False
                    self.pub_last_is_forward.publish(msg)
                    msg.data = True
                    self.pub_last_is_turn.publish(msg)
                elif c == 'L':
                    msg = BoolStamped()
                    msg.data = False
                    self.pub_last_is_forward.publish(msg)
                    msg.data = False
                    self.pub_last_is_turn.publish(msg)
                else:
                    self.stack.append('B')
                    print "Bottom of stack"
                    time.sleep(2)	    
            else:
                print "Wrong State"

            self.printroute()

        if not self.active:
            msg = BoolStamped()
            msg.data = False
            self.pub_last_is_forward.publish(msg)
            self.pub_last_is_turn.publish(msg)
            print "#####################Project not active######################"

    def printroute(self):
        route = ""
        for i in self.stack:
            if i != 'B':
                route += i
        print route

    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data
 
    def onShutdown(self):
        rospy.loginfo("[Project] Shutdown.")

if __name__ == '__main__': 
    rospy.init_node('Project',anonymous=False)
    lane_filter_node = Project()
    rospy.spin()

