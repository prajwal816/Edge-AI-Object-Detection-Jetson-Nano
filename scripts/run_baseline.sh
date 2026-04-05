#!/bin/bash
# ============================================================
# Edge AI Object Detection System
# Run Python Baseline
# ============================================================

PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Running Python Baseline Benchmark..."
export PYTHONPATH=$(pwd)
$PYTHON_CMD benchmarks/benchmark_runner.py "$@"

if [ $? -eq 0 ]; then
    echo "Comparing with C++ pipeline..."
    $PYTHON_CMD benchmarks/compare_pipelines.py
fi
