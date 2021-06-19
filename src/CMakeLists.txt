project(libs)

#external libraries used in Code 
link_libraries(-lcusolver -lcudart) # include cuda
link_libraries(-lmgl-qt5 -lmgl) # Plotter mathgl
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


#sources
aux_source_directory(. SRCS )
aux_source_directory(./Plotter PLOTTER)
aux_source_directory(./Cuda CUDA)
aux_source_directory(./Equation_System SYSTEM)
aux_source_directory(./LinearSolver LINSOL)

add_library(${PROJECT_NAME} SHARED ${SRCS}
            ${PLOTTER}
            ${CUDA}
            ${SYSTEM}
            ${LINSOL}
            )

target_include_directories(${PROJECT_NAME} PUBLIC .)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)