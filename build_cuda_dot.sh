#!/bin/bash
# Compilation script for BSI dot product CUDA acceleration

# Function to find CUDA installation
find_cuda() {
    # Try common locations
    for path in \
        "/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/targets/x86_64-linux" \
        "/usr/local/cuda" \
        "/opt/cuda" \
        "/usr/cuda" \
        "/Developer/NVIDIA/CUDA-*"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

# Find CUDA path
CUDA_PATH=$(find_cuda)

# Check if CUDA path was found
if [ -z "$CUDA_PATH" ]; then
    echo "Warning: CUDA installation not found. Will try to compile without CUDA."
    HAS_CUDA=0
else
    echo "Using CUDA installation at: $CUDA_PATH"
    HAS_CUDA=1
    
    # Add CUDA to PATH
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    
    # Check if nvcc is available
    if ! command -v nvcc &> /dev/null; then
        echo "Warning: nvcc not found in PATH. Will try to compile without CUDA."
        HAS_CUDA=0
    fi
fi

# Create build directory
mkdir -p build
cd build

CPP_FLAGS="-std=c++11 -I.. -Wall"
CUDA_INCLUDE=""
CUDA_LIBS=""
CUDA_DEFINES=""

# Set up compilation flags based on CUDA availability
if [ "$HAS_CUDA" -eq 1 ]; then
    CUDA_INCLUDE="-I\"$CUDA_PATH/include\""
    CUDA_LIBS="-L\"$CUDA_PATH/lib64\" -lcudart"
    CUDA_DEFINES="-DUSE_CUDA"
    
    # Compile CUDA kernels
    echo "Compiling CUDA kernels..."
    nvcc -c ../bsi/hybridBitmap/bsi_dot_cuda.cu -o bsi_dot_cuda_kernel.o
    if [ $? -ne 0 ]; then
        echo "Error compiling CUDA kernels. Falling back to CPU-only build."
        HAS_CUDA=0
        CUDA_DEFINES=""
    fi
fi

# Compile C++ wrappers
echo "Compiling C++ wrappers..."
g++ $CPP_FLAGS $CUDA_INCLUDE $CUDA_DEFINES -c ../bsi/hybridBitmap/bsi_dot_cuda_wrapper.cpp -o bsi_dot_cuda_wrapper.o
if [ $? -ne 0 ]; then
    echo "Error compiling bsi_dot_cuda_wrapper.cpp"
    exit 1
fi

g++ $CPP_FLAGS $CUDA_INCLUDE $CUDA_DEFINES -c ../bsi/hybridBitmap/bsi_dot_cuda.cpp -o bsi_dot_cuda.o
if [ $? -ne 0 ]; then
    echo "Error compiling bsi_dot_cuda.cpp"
    exit 1
fi

# Compile benchmark
echo "Compiling benchmark..."
g++ $CPP_FLAGS $CUDA_INCLUDE $CUDA_DEFINES -c ../examples/bsi_dot_benchmark.cpp -o bsi_dot_benchmark.o
if [ $? -ne 0 ]; then
    echo "Error compiling bsi_dot_benchmark.cpp"
    exit 1
fi

# Link everything
echo "Linking..."
if [ "$HAS_CUDA" -eq 1 ]; then
    g++ $CPP_FLAGS bsi_dot_benchmark.o bsi_dot_cuda.o bsi_dot_cuda_wrapper.o bsi_dot_cuda_kernel.o $CUDA_LIBS -o bsi_dot_benchmark
else
    g++ $CPP_FLAGS bsi_dot_benchmark.o bsi_dot_cuda.o bsi_dot_cuda_wrapper.o -o bsi_dot_benchmark
fi

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. You can run the benchmark with: ./bsi_dot_benchmark"
else
    echo "Compilation failed."
    exit 1
fi
