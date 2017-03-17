#!/bin/bash

#echo "Open LCM Tunnel"
#bot-lcm-tunnel 10.0.1.5 &
#echo "...done."

echo "PiCamera ROS to LCM"
roslaunch mocap camera.launch &&
echo "...done."
