#!/bin/bash
# ============================================================
# Edge AI Object Detection System
# Run C++ Pipeline
# ============================================================

BUILD_DIR="build"
EXE="${BUILD_DIR}/jetson_detector"

if [ ! -f "$EXE" ]; then
    echo "Error: Executable not found. Run ./scripts/build.sh first."
    exit 1
fi

MODE=${1:-"demo"}

case $MODE in
    "demo")
        echo "Running Demo Mode (Synthetic Camera + Display)"
        $EXE --source synthetic --display --precision fp16
        ;;
    "benchmark")
        echo "Running Benchmark Mode"
        $EXE --source synthetic --benchmark --frames 300 --no-display --precision fp16
        ;;
    "live")
        echo "Running Live Camera Mode (USB)"
        $EXE --source usb --precision fp16
        ;;
    "csi")
        echo "Running Live Camera Mode (Jetson CSI)"
        $EXE --source csi --precision fp16
        ;;
    "help")
        $EXE --help
        ;;
    *)
        echo "Unknown mode. Using custom arguments."
        $EXE "$@"
        ;;
esac
