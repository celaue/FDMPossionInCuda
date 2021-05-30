cmake_minimum_required(VERSION 3.10)
project(FDM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("include")


add_subdirectory(System)



add_executable(FDM main.cpp)
target_link_libraries(FDM PUBLIC System)
