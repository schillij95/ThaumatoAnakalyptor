cmake_minimum_required(VERSION 3.4)
project(sheet_generation)

# Set the build type if it's not already set
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# Add pybind11 submodule
add_subdirectory(pybind11)

# Add YAML-CPP submodule
add_subdirectory(yaml-cpp)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)  # Find the threading library

pybind11_add_module(sheet_generation solver.cpp)
pybind11_add_module(evolve_graph evolutionary_solve.cpp)
# Link YAML-CPP library
target_link_libraries(sheet_generation PRIVATE yaml-cpp Eigen3::Eigen ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(evolve_graph PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# add_executable(evolutionary_solve evolutionary_solve.cpp)  # Include the main entry point and source files
# target_link_libraries(evolutionary_solve PRIVATE yaml-cpp Eigen3::Eigen ${CMAKE_THREAD_LIBS_INIT})  # Link pthread and other libraries

