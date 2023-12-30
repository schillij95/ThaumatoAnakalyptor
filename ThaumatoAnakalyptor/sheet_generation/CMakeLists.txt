cmake_minimum_required(VERSION 3.4)
project(sheet_generation)

# Set the build type if it's not already set
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

# Add pybind11 submodule
add_subdirectory(pybind11)

# Add YAML-CPP submodule
add_subdirectory(yaml-cpp)

pybind11_add_module(sheet_generation solver.cpp)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Link YAML-CPP library
target_link_libraries(sheet_generation PRIVATE yaml-cpp Eigen3::Eigen)

