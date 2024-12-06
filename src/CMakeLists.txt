cmake_minimum_required(VERSION 3.23)
project(HighPerformanceLinearAlgebra VERSION 1.0)

# Set the include directory so can cleanly include .hpp files
include_directories(${PROJECT_SOURCE_DIR}/../include)
# Update include path to use external/mdspan
include_directories(${PROJECT_SOURCE_DIR}/../external/mdspan/include)

# Set to C++ 23, since we will be using packages only in C++ 23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Add optimization flags for builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Define a variable for the exercise number, defaulting to 1 if not set
set(EXERCISE_NUMBER 1 CACHE STRING "Exercise number to build")

# Set the source file name to Exercise#.cpp
set(SOURCE_FILE "Exercise${EXERCISE_NUMBER}.cpp")
message("Configuring build for: ${SOURCE_FILE}")

# Create the executable file for this exercise given the proper source file
add_executable(Exercise${EXERCISE_NUMBER} ${SOURCE_FILE})

# Add OpenMP package for Intel compilation
find_package(OpenMP REQUIRED)
add_compile_options(-fopenmp)
target_compile_options(Exercise${EXERCISE_NUMBER} PRIVATE -fopenmp)
# Link OpenMP runtime to the executable
target_link_libraries(Exercise${EXERCISE_NUMBER} PRIVATE OpenMP::OpenMP_CXX)

# Ensure target compiled with C++23
target_compile_features(Exercise${EXERCISE_NUMBER} PRIVATE cxx_std_23)