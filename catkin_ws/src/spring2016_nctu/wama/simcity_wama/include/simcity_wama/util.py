# Import stuff here
import random
import rospy
import os
import yaml
import numpy as np
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
        array_index = 0; # causes only one markerarray to be drawn?
        return self.get_tile_message(tile_name, tile_center_position, array_index)

    def get_tile_message(self, tile_name, tile_center_position, array_index):
        # according to args, construct and return the appropriate tile message (for now, a simple marker array message)

        tile_info = self.tile_dict[tile_name] #for now, will include the east_west//north_south and number of lanes at the end. might switch to id # later?

        tile_length = 1.0 # TODO rmata pull info from yaml? some config file? leave it as 1 for now
        tile_origin_position = (np.array(tile_center_position) 
                                - np.array([tile_length/2, tile_length/2])).tolist() 
        # lower leftmost corner of tile is the origin (convention)
        
        # construct the tile message
        tilemsg = MarkerArray()
        arrow_ids = range(len(tile_info['connectivity']))
        i = 0
        for lane in tile_info['connectivity']:
            # calculate start and end of arrow, as per lane
            arrow_start = (np.array(tile_info['nodes_positions'][lane[0]])*tile_length 
                           + np.array(tile_origin_position)).tolist()
            arrow_end = (np.array(tile_info['nodes_positions'][lane[1]])*tile_length 
                         + np.array(tile_origin_position)).tolist()
            # add the appropriate marker
            arrow = Marker()
            arrow.type = 0 # arrow
            arrow.action = 0 # add
            arrow.header.frame_id  = "/map"
            arrow.id = arrow_ids[i]+8*array_index; i+=1

            # scales of arrow
            arrow.scale.x = 0.025
            arrow.scale.y = 0.1
            arrow.scale.z = 0.1
            # start, end?
            arrow.points.append(Point(arrow_start[0], arrow_start[1], 0))
            arrow.points.append(Point(arrow_end[0], arrow_end[1], 0))
            # color
            arrow.color.r = 1.0
            arrow.color.b = 1.0
            arrow.color.g = 0.0
            arrow.color.a = 1.0
            tilemsg.markers.append(arrow)
       
        return tilemsg

if __name__ == '__main__':
    # Test code for ModuleNames
    print 'No test code for simcity in main yet.'

