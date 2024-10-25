cmake_minimum_required(VERSION 3.18)

# Project name and version
project(graph_problem VERSION 1.0 LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Enable or disable GPU support
option(ENABLE_GPU "Enable GPU compilation" ON)

# Find and enable OpenMP
find_package(OpenMP REQUIRED)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# ------------------------ CPU Version ----------------------------------
# Add the executable for the CPU version
add_executable(graph_problem main.cpp)

# Include the directory where argparse.hpp is located for the CPU version
target_include_directories(graph_problem PRIVATE ${CMAKE_SOURCE_DIR})

# Link OpenMP and OpenCV libraries for the CPU version
target_link_libraries(graph_problem PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})

# Add compile options for the CPU version
target_compile_options(graph_problem PRIVATE
    $<$<CONFIG:Release>:-O3>       # Use -O3 for Release builds
    $<$<CONFIG:Debug>:-O0>         # No optimization for Debug builds
    -Wall -Wextra -pedantic        # Warnings for both builds
)

# ------------------------ GPU Version ----------------------------------
if(ENABLE_GPU)
    # Find CUDA toolkit (recommended for modern CMake)
    find_package(CUDAToolkit REQUIRED)

    # Detect architecture if not explicitly defined
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Default to common architectures if automatic detection fails
        set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80)
    endif()

    # Add the executable for the GPU version
    add_executable(graph_problem_gpu main_gpu.cpp solve_gpu.cu)

    # Include the directory where argparse.hpp and other headers are located for the GPU version
    target_include_directories(graph_problem_gpu PRIVATE ${CMAKE_SOURCE_DIR})

    # Link OpenMP, OpenCV, and CUDA runtime libraries for the GPU version
    target_link_libraries(graph_problem_gpu PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS} CUDA::cudart)

    # Add compile options for the GPU version
    target_compile_options(graph_problem_gpu PRIVATE
        $<$<CONFIG:Release>:-use_fast_math>  # Enable fast math for faster performance (if precision allows)
        $<$<CONFIG:Release>:-O3>       # Use -O3 for Release builds
        $<$<CONFIG:Debug>:-O0>         # No optimization for Debug builds
    )

    # Set CUDA architectures explicitly for GPU version
    set_target_properties(graph_problem_gpu PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON  # Required if you plan to link shared libraries
    )
endif()
