#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import rospy
import pigpio
import time
from dht_sean.msg import Sensor

class DHT(object):
    def __init__(self,pi,gpio):
	self.pi=pi
	self.gpio=gpio
	self.high_tick=0
	self.bit=40
	self.temperature=0
	self.humidity=0
	self.either_edge_cb=None
	self.setup()
	self.node_name=rospy.get_name()
	self.pub_timestep=self.setupParam("~pub_timestep",1)
	self.pub_dht=rospy.Publisher("~dht_data",Sensor,queue_size=10)
	self.pub_timer=rospy.Timer(rospy.Duration.from_sec(self.pub_timestep),self.publish)
	rospy.loginfo("[%s] Initializing " %(self.node_name))		
    def setupParam(self,param_name,default_value):
	value=rospy.get_param(param_name,default_value)
	rospy.set_param(param_name,value)
	rospy.loginfo("[%s] %s = %s" %(self.node_name,param_name,value))
	return value		
    def publish(self,event):
	sensor_data=Sensor()
	sensor_data.temperature=self.temperature
	sensor_data.moisture=self.humidity
	self.pub_dht.publish(sensor_data)	
    def onShutdown(self):
	rospy.loginfo("[%s] Shutting down." %(self.node_name))		
    def setup(self):
	self.pi.set_pull_up_down(self.gpio,pigpio.PUD_OFF)
	self.pi.set_watchdog(self.gpio,0)
	self.register_callbacks()
    def register_callbacks(self):
	self.either_edge_cb=self.pi.callback(self.gpio,pigpio.EITHER_EDGE,self.either_edge_callback)
    def either_edge_callback(self,gpio,level,tick):
	level_handlers={pigpio.FALLING_EDGE:self._edge_FALL,pigpio.RISING_EDGE:self._edge_RISE,pigpio.EITHER_EDGE:self._edge_EITHER}
	handler=level_handlers[level]
	diff = pigpio.tickDiff(self.high_tick, tick)
	handler(tick, diff)
    def _edge_RISE(self, tick, diff):
	val = 0
	if diff >= 50:
	    val = 1
	if diff >= 200:
	    self.checksum = 256
	if self.bit >= 40:
	    self.bit = 40
	elif self.bit >= 32:
	    self.checksum = (self.checksum << 1) + val
	    if self.bit == 39:
		self.pi.set_watchdog(self.gpio, 0)
		total = self.humidity + self.temperature
		if not (total & 255) == self.checksum:
		    raise
	elif 16 <= self.bit < 24:
	    self.temperature = (self.temperature << 1) + val
	elif 0 <= self.bit < 8:
	    self.humidity = (self.humidity << 1) + val
	else:
	    pass
	self.bit+=1	
    def _edge_FALL(self, tick, diff):
	self.high_tick = tick
	if diff <= 250000:
	    return 
	self.bit=-2
	self.checksum=0
	self.temperature=0
	self.humidity=0
    def _edge_EITHER(self, tick, diff):
	self.pi.set_watchdog(self.gpio, 0)
    def read(self):
	self.pi.write(self.gpio, pigpio.LOW)
	time.sleep(0.017)
	self.pi.set_mode(self.gpio, pigpio.INPUT)
	self.pi.set_watchdog(self.gpio, 200)
	time.sleep(0.2)
    def close(self):
	self.pi.set_watchdog(self.gpio, 0)
	if self.either_edge_cb:
	    self.either_edge_cb.cancel()
	    self.either_edge_cb = None
    def __iter__(self):
	return self
    def next(self):
	self.read()
	response={'humidity':self.humidity,'temperature':self.temperature}
	return response
if __name__=='__main__':
    rospy.init_node("Dht_sean",anonymous=False)
    pi=pigpio.pi()
    sensor=DHT(pi,4)
    rospy.on_shutdown(sensor.onShutdown)
    rospy.spin()
    sensor.close()
