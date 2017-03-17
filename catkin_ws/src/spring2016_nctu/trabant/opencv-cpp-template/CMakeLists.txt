# cmake_minimum_required(VERSION 2.6)
cmake_minimum_required(VERSION 2.8.3)
project(opencv_cpp_template)

#add_definitions(-pg) #"-fopenmp)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  # message_generation
  duckietown_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  tf_conversions
  tf
)

find_package(OpenCV REQUIRED)
# find_package(eigen3 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES apriltags
 # CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs visualization_msgs #acl_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# file(GLOB SOURCE_FILES "src/*.cc")
# add_library(apriltags ${SOURCE_FILES})
# add_dependencies(apriltags ${catkin_EXPORTED_TARGETS})
# target_link_libraries(apriltags ${OpenCV_LIBS}) #-pg) #-fopenmp)

add_executable(opencv_cpp_template_node src/opencv_cpp_template_node.cpp)
add_dependencies(opencv_cpp_template_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(opencv_cpp_template_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})
# target_link_libraries(apriltags ${OpenCV_LIBS}) #-pg) #-fopenmp)
# pods_use_pkg_config_packages(apriltags eigen3)

# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#   target_link_libraries(apriltags -L/opt/local/lib/) # MacPorts special treatment...
# else (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#   pods_use_pkg_config_packages(apriltags libv4l2)
# endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# pods_install_libraries(apriltags)

# file(GLOB header_files "AprilTags/*.h")
# pods_install_headers(${header_files} DESTINATION AprilTags/)

# pods_install_pkg_config_file(apriltags
#     LIBS -lapriltags
#     REQUIRES eigen3 opencv2
#     VERSION 1.0.0)

# add_subdirectory(example)
