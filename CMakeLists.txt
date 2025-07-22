cmake_minimum_required(VERSION 3.17)
project(IOWCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(OpenGL REQUIRED)

include_directories("include")
link_directories(lib-vc2019)
add_executable(IOWCuda kernel.cu lodepng.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")

set_target_properties(
        IOWCuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(IOWCuda glew32.lib glew32s.lib glfw3.lib glfw3.dll glfw3dll.lib OpenGL::GL)
