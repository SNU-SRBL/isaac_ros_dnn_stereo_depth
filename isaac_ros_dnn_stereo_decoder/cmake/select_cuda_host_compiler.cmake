# CUDA 12.1 rejects GCC 13. Select GCC 12 before project(... CUDA) enables nvcc.
find_program(ISAAC_ROS_GCC_12 NAMES gcc-12 REQUIRED)
find_program(ISAAC_ROS_GXX_12 NAMES g++-12 REQUIRED)

set(CMAKE_C_COMPILER "${ISAAC_ROS_GCC_12}" CACHE FILEPATH
  "C compiler for isaac_ros_dnn_stereo_decoder" FORCE)
set(CMAKE_CXX_COMPILER "${ISAAC_ROS_GXX_12}" CACHE FILEPATH
  "C++ compiler for isaac_ros_dnn_stereo_decoder" FORCE)
set(CMAKE_CUDA_HOST_COMPILER "${ISAAC_ROS_GXX_12}" CACHE FILEPATH
  "CUDA host compiler for isaac_ros_dnn_stereo_decoder" FORCE)
