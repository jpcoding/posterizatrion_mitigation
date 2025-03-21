
cmake_minimum_required(VERSION 3.12)
project(PM LANGUAGES CXX )
set (CMAKE_CXX_STANDARD 17)


# add current include directory 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
#include sz3 
include_directories("/scratch/pji228/gittmp/sz3.1/include")
add_subdirectory(test)
add_subdirectory(test_mpi)