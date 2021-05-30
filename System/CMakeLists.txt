project(System)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_library(System System.cpp)
target_link_libraries (System Eigen3::Eigen)