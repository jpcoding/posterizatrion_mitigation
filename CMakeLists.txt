cmake_minimum_required(VERSION 3.12)
project(PM LANGUAGES CXX )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
#include sz3 
include_directories("/scratch/pji228/gittmp/sz3.1/include/")
add_subdirectory(test)
add_subdirectory(test_mpi)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)