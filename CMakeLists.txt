cmake_minimum_required(VERSION 3.18)
project(JetsonEdgeAI 
    VERSION 1.0.0
    DESCRIPTION "Edge AI Object Detection System for NVIDIA Jetson Nano"
    LANGUAGES CXX
)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ============================================================================
# Options
# ============================================================================
option(SIMULATE_GPU "Enable GPU simulation mode (no CUDA/TensorRT required)" ON)
option(USE_FP16 "Enable FP16 inference precision" ON)
option(USE_INT8 "Enable INT8 inference precision" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_PROFILING "Enable pipeline profiling" ON)

# ============================================================================
# Find Dependencies
# ============================================================================

# OpenCV (required)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Threads (required)
find_package(Threads REQUIRED)

# CUDA (optional - auto-detected)
if(NOT SIMULATE_GPU)
    find_package(CUDA)
    if(CUDA_FOUND)
        message(STATUS "CUDA version: ${CUDA_VERSION}")
        enable_language(CUDA)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
    else()
        message(WARNING "CUDA not found. Falling back to simulation mode.")
        set(SIMULATE_GPU ON)
    endif()
endif()

# TensorRT (optional - auto-detected)
if(NOT SIMULATE_GPU)
    find_library(TENSORRT_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib)
    find_path(TENSORRT_INCLUDE NvInfer.h HINTS /usr/include/aarch64-linux-gnu /usr/local/include)
    if(TENSORRT_LIB AND TENSORRT_INCLUDE)
        message(STATUS "TensorRT found: ${TENSORRT_LIB}")
        set(TENSORRT_FOUND TRUE)
    else()
        message(WARNING "TensorRT not found. Falling back to simulation mode.")
        set(SIMULATE_GPU ON)
    endif()
endif()

if(SIMULATE_GPU)
    message(STATUS "=== SIMULATION MODE ENABLED ===")
    message(STATUS "  CUDA and TensorRT calls will be simulated")
    add_definitions(-DSIMULATE_GPU)
endif()

if(USE_FP16)
    add_definitions(-DUSE_FP16)
endif()

if(USE_INT8)
    add_definitions(-DUSE_INT8)
endif()

if(ENABLE_PROFILING)
    add_definitions(-DENABLE_PROFILING)
endif()

# ============================================================================
# Source Files
# ============================================================================
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})

set(INFERENCE_SOURCES
    ${PROJECT_ROOT}/cpp/inference/trt_engine.cpp
)

set(PIPELINE_SOURCES
    ${PROJECT_ROOT}/cpp/pipeline/pipeline.cpp
)

set(CUDA_UTILS_SOURCES
    ${PROJECT_ROOT}/cpp/cuda_utils/gpu_memory_pool.cpp
    ${PROJECT_ROOT}/cpp/cuda_utils/preprocessor.cpp
)

set(CAMERA_SOURCES
    ${PROJECT_ROOT}/cpp/camera/camera_capture.cpp
)

set(ALL_SOURCES
    ${INFERENCE_SOURCES}
    ${PIPELINE_SOURCES}
    ${CUDA_UTILS_SOURCES}
    ${CAMERA_SOURCES}
)

# ============================================================================
# Include Directories
# ============================================================================
set(PROJECT_INCLUDES
    ${PROJECT_ROOT}/cpp
    ${OpenCV_INCLUDE_DIRS}
)

if(TENSORRT_FOUND)
    list(APPEND PROJECT_INCLUDES ${TENSORRT_INCLUDE} ${CUDA_INCLUDE_DIRS})
endif()

# ============================================================================
# Main Executable
# ============================================================================
add_executable(jetson_detector
    ${PROJECT_ROOT}/cpp/main.cpp
    ${ALL_SOURCES}
)

target_include_directories(jetson_detector PRIVATE ${PROJECT_INCLUDES})
target_link_libraries(jetson_detector PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

if(TENSORRT_FOUND)
    target_link_libraries(jetson_detector PRIVATE
        ${TENSORRT_LIB}
        ${CUDA_LIBRARIES}
        nvinfer_plugin
        nvonnxparser
    )
endif()

# ============================================================================
# Test Executables
# ============================================================================
if(BUILD_TESTS)
    # Inference unit test
    add_executable(test_inference
        ${PROJECT_ROOT}/tests/test_inference.cpp
        ${INFERENCE_SOURCES}
        ${CUDA_UTILS_SOURCES}
    )
    target_include_directories(test_inference PRIVATE ${PROJECT_INCLUDES})
    target_link_libraries(test_inference PRIVATE
        ${OpenCV_LIBS}
        Threads::Threads
    )

    # Pipeline integration test
    add_executable(test_pipeline
        ${PROJECT_ROOT}/tests/test_pipeline.cpp
        ${ALL_SOURCES}
    )
    target_include_directories(test_pipeline PRIVATE ${PROJECT_INCLUDES})
    target_link_libraries(test_pipeline PRIVATE
        ${OpenCV_LIBS}
        Threads::Threads
    )

    if(TENSORRT_FOUND)
        target_link_libraries(test_inference PRIVATE ${TENSORRT_LIB} ${CUDA_LIBRARIES})
        target_link_libraries(test_pipeline PRIVATE ${TENSORRT_LIB} ${CUDA_LIBRARIES})
    endif()

    # CTest integration
    enable_testing()
    add_test(NAME InferenceTest COMMAND test_inference)
    add_test(NAME PipelineTest COMMAND test_pipeline)
endif()

# ============================================================================
# Install Rules
# ============================================================================
install(TARGETS jetson_detector RUNTIME DESTINATION bin)
install(DIRECTORY ${PROJECT_ROOT}/configs/ DESTINATION etc/jetson_detector)
install(DIRECTORY ${PROJECT_ROOT}/models/ DESTINATION share/jetson_detector/models
    FILES_MATCHING PATTERN "*.engine" PATTERN "*.onnx")

# ============================================================================
# Build Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Jetson Edge AI - Build Configuration")
message(STATUS "========================================")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:    C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Simulation:      ${SIMULATE_GPU}")
message(STATUS "  FP16:            ${USE_FP16}")
message(STATUS "  INT8:            ${USE_INT8}")
message(STATUS "  Build tests:     ${BUILD_TESTS}")
message(STATUS "  Profiling:       ${ENABLE_PROFILING}")
message(STATUS "  OpenCV:          ${OpenCV_VERSION}")
if(CUDA_FOUND)
    message(STATUS "  CUDA:            ${CUDA_VERSION}")
endif()
if(TENSORRT_FOUND)
    message(STATUS "  TensorRT:        Found")
endif()
message(STATUS "========================================")
message(STATUS "")
