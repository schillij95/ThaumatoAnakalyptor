cmake_minimum_required(VERSION 3.10)

# Project name and version
project(graph_problem VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Find and enable OpenMP
find_package(OpenMP REQUIRED)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(graph_problem main.cpp)

# Include the directory where argparse.hpp is located
target_include_directories(graph_problem PRIVATE ${CMAKE_SOURCE_DIR})

# Link OpenMP and OpenCV libraries
target_link_libraries(graph_problem PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})

# Add compile options
# Set the optimization level to -O3 for release builds
target_compile_options(graph_problem PRIVATE
    $<$<CONFIG:Release>:-O3>       # Use -O3 for Release builds
    $<$<CONFIG:Debug>:-O0>         # No optimization for Debug builds
    -Wall -Wextra -pedantic        # Warnings for both builds
)