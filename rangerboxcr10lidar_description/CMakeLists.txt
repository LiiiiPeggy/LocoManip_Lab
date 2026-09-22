cmake_minimum_required(VERSION 2.8.3)

# ################################
# CMake: package renamed to rangerboxcr10lidar_description begin
# ################################
project(rangerboxcr10lidar_description)
# ################################
# CMake: package renamed to rangerboxcr10lidar_description end
# ################################

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
