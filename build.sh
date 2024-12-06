#!/bin/bash

# If the argument for the (exercise number) is not passed, throw error
if [ -z "$1" ]; then
    echo "Error: Please specify the exercise number (e.g., ./build.sh 1)"
    exit 1
fi

# Set Exercise Number to the input, to be passed to CMakeLists.txt
EXERCISE_NUMBER=$1

# If the source file does not exist, throw error
if [ ! -f "src/Exercise${EXERCISE_NUMBER}.cpp" ]; then 
    echo "Error: Exercise${EXERCISE_NUMBER}.cpp does not exist!"
    exit 1
fi

# Set number of threads to 4, since our recursive approach uses 4 threads
OMP_NUM_THREADS=4

# Create (if it doesn't already exist) or clean the build directory
mkdir -p build
# Move into the build directory for CMake to begin
cd build

# Run CMake with the specified Exercise Number
cmake ../src -DEXERCISE_NUMBER=$EXERCISE_NUMBER

# Build the project
make

# Run the newly compiled executable
./Exercise$EXERCISE_NUMBER

# MAKE SURE TO MAKE THE SCRIPT EXECUTABLE WITH
# chmod +x build.sh
