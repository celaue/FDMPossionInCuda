cmake_minimum_required(VERSION 3.10)
project(fdm LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_subdirectory(src/)
aux_source_directory(src SRCS)
add_executable(${PROJECT_NAME} ${SRCS})

set_property(TARGET ${PROJECT_NAME}
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} PRIVATE libs)
