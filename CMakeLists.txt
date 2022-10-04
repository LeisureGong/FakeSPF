cmake_minimum_required(VERSION 3.23)
project(SPF)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(SPF
        metis.h
        test.cpp)
