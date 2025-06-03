find_cuda() {
    
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
# Use environment-provided CUDA path if available
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    CUDA_PATH="$CUDA_HOME"
elif [ -n "$CUDA_ROOT" ] && [ -d "$CUDA_ROOT" ]; then
    CUDA_PATH="$CUDA_ROOT"
else
    CUDA_PATH=$(find_cuda)
fi

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

CPP_FLAGS="-std=c++17 -I.. -Wall -Wno-error=reorder"
CUDA_INCLUDE=""
CUDA_LIBS=""
CUDA_DEFINES=""

# Set up compilation flags based on CUDA availability
if [ "$HAS_CUDA" -eq 1 ]; then
    CUDA_INCLUDE="-I$CUDA_PATH/include"

    # Try to locate libcudart in common subdirectories
    CUDA_LIB_PATH=""
    for libdir in "$CUDA_PATH/lib64" "$CUDA_PATH/lib" "$CUDA_PATH/targets/x86_64-linux/lib" "$CUDA_PATH/targets/x86_64-linux/lib64" "$CUDA_PATH/targets/x86_64-linux/lib/stubs"; do
        if [ -e "$libdir/libcudart.so" ] || [ -e "$libdir/libcudart_static.a" ]; then
            CUDA_LIB_PATH="$libdir"
            break
        fi
    done

    if [ -z "$CUDA_LIB_PATH" ]; then
        echo "Error: Could not locate libcudart.so in $CUDA_PATH. Falling back to CPU-only build."
        HAS_CUDA=0
        CUDA_DEFINES=""
    else
        CUDA_LIBS="-L$CUDA_LIB_PATH -lcudart"
        # Define USE_CUDA for compilation since CUDA is available
        CUDA_DEFINES="-DUSE_CUDA"
        # Ensure runtime loader can find the library when running executable
        export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
    fi
    
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

# Compile BSI source files that have non-header implementations (currently mostly empty but future-proof)
# These compile quickly and avoid undefined reference errors if functions move into .cpp files.

echo "Compiling BsiSigned / BsiUnsigned (if any)..."
g++ $CPP_FLAGS -c ../bsi/BsiSigned.cpp -o BsiSigned.o
if [ $? -ne 0 ]; then
    echo "Error compiling BsiSigned.cpp"; exit 1; fi

g++ $CPP_FLAGS -c ../bsi/BsiUnsigned.cpp -o BsiUnsigned.o
if [ $? -ne 0 ]; then
    echo "Error compiling BsiUnsigned.cpp"; exit 1; fi

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
    g++ $CPP_FLAGS bsi_dot_benchmark.o bsi_dot_cuda.o bsi_dot_cuda_wrapper.o bsi_dot_cuda_kernel.o BsiSigned.o BsiUnsigned.o $CUDA_LIBS -o bsi_dot_benchmark
else
    g++ $CPP_FLAGS bsi_dot_benchmark.o bsi_dot_cuda.o bsi_dot_cuda_wrapper.o BsiSigned.o BsiUnsigned.o -o bsi_dot_benchmark
fi

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. You can run the benchmark with: ./bsi_dot_benchmark"
else
    echo "Compilation failed."
    exit 1
fi
