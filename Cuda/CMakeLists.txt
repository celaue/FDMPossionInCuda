cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
link_directories(/usr/local/cuda/lib64)
link_libraries(-lcusolver -lcudart)

add_executable(particle_test ParallelSolver.cu)
 
set_property(TARGET particle_test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries (particle_test Eigen3::Eigen)
