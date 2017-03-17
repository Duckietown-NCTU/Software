# Import stuff here
import random
import rospy
import os
import yaml
import numpy as np
import math
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

# Implement modules
class TileProduction(object):
    def __init__(self, mapfile, tilesfile):
        ''' 
        Initialize dictionary of tiles, perhaps with tile messages. 
        Initialize map of tiles.
        Initialize index at which the production line is currently
         at (ie which next tile of map must be sent).
        '''
        dict_file = open(tilesfile, 'r')
        self.tile_dict = yaml.load(dict_file)
        dict_file.close()
        
        map_file = open(mapfile, 'r') 
        self.map_list = yaml.load(map_file)
        map_file.close()
        
        self.num_tiles_in_map = len(self.map_list)

        self.next_map_index = 0 # next index of tile to send
        self.point1=[0,0]
        self.point2=[0,0]
        self.pontt11=[0,0]
        self.point22=[0,0]
        self.rotate1=[]
        self.rotate2=[]
    def get_num_tiles_in_map(self):
        return self.num_tiles_in_map

    def get_certain_tile_message(self, array_index):
        '''
        Gets tile_index tile from the map, puts it into markerArray message format.
        Arg array_index MUST be within the bounds of map_list 
         when get_certain_tile_message is called.
        '''
        next_tile_info = self.map_list[array_index]
        tile_name = next_tile_info['tile_type']
        tile_center_position = next_tile_info['tile_center']
        
        return self.get_tile_message(tile_name, tile_center_position, array_index)

    def get_next_tile_message(self):
        '''
        Gets next tile from the map.
        '''
        next_tile_info = self.map_list[self.next_map_index]
        if self.next_map_index >= len(next_tile_info):
            return MarkerArray() # should be empty marker? end stream? what to do?
        tile_name = next_tile_info['tile_type']
        tile_center_position = next_tile_info['tile_center']
        
        self.next_map_index += 1
        array_index = 1; # causes only one markerarray to be drawn?
        return self.get_tile_message(tile_name, tile_center_position, array_index)

    def get_tile_message(self, tile_name, tile_center_position, array_index):
        # according to args, construct and return the appropriate tile message (for now, a simple marker array message)
        tile_info = self.tile_dict[tile_name] #for now, will include the east_west//north_south and number of lanes at the end. might switch to id # later?

        tile_length = 1 # TODO rmata pull info from yaml? some config file? leave it as 1 for now
        tile_origin_position = (np.array(tile_center_position) 
                                - np.array([tile_length/2, tile_length/2])).tolist() 
        # lower leftmost corner of tile is the origin (convention)
        print tile_origin_position 
        # construct the tile message
        tilemsg = MarkerArray()
        arrow_ids = range(len(tile_info['rec_white']))
        j = 0
        for lane in tile_info['rec_white']:
            # calculate start and end of arrow, as per lane
            print lane
            self.point11 = (np.array(tile_info['nodes_positions'][lane[0]])*tile_length 
                           ).tolist()
            self.point22 = (np.array(tile_info['nodes_positions'][lane[1]])*tile_length
                           ).tolist()
            tarrow = Marker() #brian
            tarrow.type = 1
            tarrow.action = 0
            tarrow.header.frame_id  = "/map"
            tarrow.id = array_index*8+j+1000
            j+=1
            
            self.rotate1=np.array(tile_info['nodes_positions'][lane[2]]).tolist()
            self.rotate2=np.array(tile_info['nodes_positions'][lane[3]]).tolist()
            print self.rotate1
            print self.rotate2
            print self.point11
            print self.point22
            self.point1[0]=self.rotate1[0]*self.point11[0]+self.rotate1[1]*self.point11[1]+self.rotate1[2]
            self.point1[1]=self.rotate2[0]*self.point11[0]+self.rotate2[1]*self.point11[1]+self.rotate2[2]
            self.point2[0]=self.rotate1[0]*self.point22[0]+self.rotate1[1]*self.point22[1]+self.rotate1[2]
            self.point2[1]=self.rotate2[0]*self.point22[0]+self.rotate2[1]*self.point22[1]+self.rotate2[2]
            print self.point1
            print self.point2
            print tile_origin_position 
            tarrow.scale.x = self.point2[0]-self.point1[0]
            tarrow.scale.y = self.point2[1]-self.point1[1]
            tarrow.scale.z = 0
            tarrow.color.r = 255.0
            tarrow.color.b = 255.0
            tarrow.color.g = 255.0
            tarrow.color.a = 1.0
            tarrow.pose.position.x = (self.point2[0]+self.point1[0]+tile_origin_position[0]*2)/2
            tarrow.pose.position.y = (self.point2[1]+self.point1[1]+tile_origin_position[1]*2)/2
            tarrow.pose.position.z = 0
            tarrow.pose.orientation.x = 0.0
            tarrow.pose.orientation.y = 0.0
            tarrow.pose.orientation.z = 0.0
            #tarrow.pose.orientation.z = np.array(tile_info['nodes_positions'][lane[2]])
            tarrow.pose.orientation.w = 1.0
            print tarrow.pose.position.x
            print tarrow.pose.position.y
            tilemsg.markers.append(tarrow)
        for lane in tile_info['rec_yellow']:
            # calculate start and end of arrow, as per lane
            self.point11 = (np.array(tile_info['nodes_positions'][lane[0]])*tile_length).tolist()
            self.point22 = (np.array(tile_info['nodes_positions'][lane[1]])*tile_length).tolist()
            yarrow = Marker() #brian
            yarrow.type = 1
            yarrow.action = 0
            yarrow.header.frame_id  = "/map"
            yarrow.id = array_index*8+j+2000
            j+=1
            self.rotate1=np.array(tile_info['nodes_positions'][lane[2]]).tolist()
            self.rotate2=np.array(tile_info['nodes_positions'][lane[3]]).tolist()
            self.point1[0]=self.rotate1[0]*self.point11[0]+self.rotate1[1]*self.point11[1]+self.rotate1[2]
            self.point1[1]=self.rotate2[0]*self.point11[0]+self.rotate2[1]*self.point11[1]+self.rotate2[2]
            self.point2[0]=self.rotate1[0]*self.point22[0]+self.rotate1[1]*self.point22[1]+self.rotate1[2]
            self.point2[1]=self.rotate2[0]*self.point22[0]+self.rotate2[1]*self.point22[1]+self.rotate2[2]

            
            yarrow.scale.x = self.point2[0]-self.point1[0]
            yarrow.scale.y = self.point2[1]-self.point1[1]
            yarrow.scale.z = 0
            yarrow.color.r = 255.0
            yarrow.color.b = 0.0
            yarrow.color.g = 255.0
            yarrow.color.a = 1.0
            yarrow.pose.position.x = (self.point2[0]+self.point1[0]+tile_origin_position[0]*2)/2
            yarrow.pose.position.y = (self.point2[1]+self.point1[1]+tile_origin_position[1]*2)/2
            yarrow.pose.position.z = 0
            yarrow.pose.orientation.x = 0.0
            yarrow.pose.orientation.y = 0.0
            yarrow.pose.orientation.z = 0.0
            yarrow.pose.orientation.w = 1.0
            tilemsg.markers.append(yarrow)
#/brian



        return tilemsg

if __name__ == '__main__':
    # Test code for ModuleNames
    print 'No test code for simcity in main yet.'

