#!/bin/bash
# ============================================================
# Edge AI Object Detection System
# Build Script (CMake)
# ============================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BUILD_DIR="build"
BUILD_TYPE="Release"
CLEAN=0
SIMULATE=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) BUILD_TYPE="Debug";;
        --clean) CLEAN=1;;
        --simulate) SIMULATE=1;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
    shift
done

echo -e "${BLUE}=== Building Jetson Edge AI Pipeline ===${NC}"
echo "Build Type: $BUILD_TYPE"

if [ $CLEAN -eq 1 ]; then
    echo -e "${BLUE}Cleaning build directory...${NC}"
    rm -rf ${BUILD_DIR}
fi

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"

if [ $SIMULATE -eq 1 ]; then
    echo -e "${BLUE}Forcing GPU Simulation Mode...${NC}"
    CMAKE_ARGS="${CMAKE_ARGS} -DSIMULATE_GPU=ON"
fi

echo -e "${BLUE}Configuring CMake...${NC}"
cmake .. ${CMAKE_ARGS}

echo -e "${BLUE}Compiling...${NC}"
make -j$(nproc)

echo -e "${GREEN}Build complete!${NC}"
echo "Binaries are located in ${BUILD_DIR}/"
ls -la jetson_detector test_inference test_pipeline
