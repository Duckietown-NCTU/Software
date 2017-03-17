# Simcity - Map Editor Version 0.1
All ./ references point to duckietown(Software)/catkin_ws/src/simcity
A good reference for duckietown packages in general is catkin_ws/src/pkg_name/howto.md


## How to run the map editor

(V0.1)

Inside ./launch is basic_map_tiler.launch. Ensure that the map file's path is correct for your machine. We start simcity, given a specific map file. We also start rviz, ROS' common gui.


## How to edit the map 

(V0.1)

The map is contained in ./maps as a YAML file. map.yaml is a small example of some circular streets, and censi_map.yaml is the map of the duckietown currently up and running.


## What am I looking at, anyway? 

(V0.1)

The magenta arrows point in the direction of traffic. These arrows are lines indicating where traffic flows.


## What else is there to do?

(V0.1)

Lots of things. This part is mostly for rmata (1/11/16)

(1) Beautify it. Arrows are straight and ugly. Roads in duckietown can be curved, have not-so-subtle lane markings, stop signs, grass, and potentially cones and duckies.....

(2) Make it interactive. MarkerArrays consist of Markers. What does an InteractiveMarker consist of?

(3) Establish consistency and validation when adding a tile to a map. This would involve:
    - checking node positions at adjacent tiles
    - multiplying the sparse lane matrices and checking that number of lanes is consistent
    - having a perhaps separate node receive messages from the interactive server, or the basic_map_tiler node, and doing these computations for each change