#!/usr/bin/env python
import rospy
import signal
import sys
from duckietown_msgs.msg import WheelsCmdStamped,Twist2DStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Header

TAG = "Tank Like Joy Mapper:"

#------------------------------- NODE TEMPLATE --------------------------------#
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


# SIGINT Signal handler
def sigint_handler(signal, frame):
        global pub
        print ""
        print TAG,"Interrupt!"
        print TAG,"Terminated"
        sys.exit(0)

if __name__ == '__main__':
    print TAG,"Started"
    # Assigning the SIGINT handler.
    signal.signal(signal.SIGINT, sigint_handler)
    # Starting the node
    rospy.init_node('tank_driver', anonymous=True)
    rospy.Subscriber("/ducktruck/joy",Joy,joy_callback)
    pub = rospy.Publisher("/ducktruck/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=10)
    rospy.spin()

    print TAG,"Terminated"
