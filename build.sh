#!/bin/bash

# MAKE SURE TO MAKE THE SCRIPT EXECUTABLE WITH
# chmod +x build.sh

# Set number of threads to 4, since our recursive approach uses 4 threads
OMP_NUM_THREADS=4

# Create (if it doesn't already exist) or clean the build directory
mkdir -p build
# Move into the build directory for CMake to begin
cd build

# Run CMake with the specified Exercise Number
cmake ../src
# Build the project
make

# Run the newly compiled executable
./hpla


