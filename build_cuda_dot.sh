#!/bin/bash
# Compilation script for BSI dot product CUDA acceleration

# Detect CUDA path
CUDA_PATH="/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/targets/x86_64-linux"
if [ ! -d "$CUDA_PATH" ]; then
    # Try common locations
    for path in /usr/local/cuda /opt/cuda /usr/cuda; do
        if [ -d "$path" ]; then
            CUDA_PATH="$path"
            break
        fi
    done
fi

# Check if CUDA path was found
if [ ! -d "$CUDA_PATH" ]; then
    echo "Error: CUDA installation not found. Please set CUDA_PATH manually in this script."
    exit 1
fi

echo "Using CUDA installation at: $CUDA_PATH"

# Create build directory
mkdir -p build
cd build

# Compile CUDA kernels
echo "Compiling CUDA kernels..."
nvcc -c bsi/hybridBitmap/bsi_dot_cuda.cu -o bsi_dot_cuda.o

# Compile C++ wrappers
echo "Compiling C++ wrappers..."
g++ -c -DUSE_CUDA -I"$CUDA_PATH/include" bsi/hybridBitmap/bsi_dot_cuda_wrapper.cpp -o bsi_dot_cuda_wrapper.o
g++ -c -DUSE_CUDA -I"$CUDA_PATH/include" bsi/hybridBitmap/bsi_dot_cuda.cpp -o bsi_dot_cuda.o

# Compile benchmark
echo "Compiling benchmark..."
g++ -c -DUSE_CUDA -I"$CUDA_PATH/include" examples/bsi_dot_benchmark.cpp -o bsi_dot_benchmark.o

# Link everything
echo "Linking..."
g++ bsi_dot_benchmark.o bsi_dot_cuda.o bsi_dot_cuda_wrapper.o bsi_dot_cuda.o -L"$CUDA_PATH/lib64" -lcudart -o bsi_dot_benchmark

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful! Run the benchmark with:"
    echo "./bsi_dot_benchmark"
else
    echo "Compilation failed."
fi
