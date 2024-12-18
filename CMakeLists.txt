
cmake_minimum_required(VERSION 3.12)
project(PM LANGUAGES CXX)


# add current include directory 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(test)