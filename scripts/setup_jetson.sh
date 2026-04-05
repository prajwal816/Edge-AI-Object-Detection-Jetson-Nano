#!/bin/bash
# ============================================================
# Edge AI Object Detection System
# Setup Jetson Nano Environment
# ============================================================
# IMPORTANT: Run this directly on the Jetson device!

if [[ $(uname -m) != "aarch64" ]]; then
    echo "Warning: Not running on ARM64 architecture."
    echo "This script is intended for the Jetson Nano."
fi

echo "Setting up Jetson Environment..."

# 1. System packages
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential cmake
sudo apt-get install -y libopencv-dev libgstreamer1.0-dev

# 2. Maximize performance
echo "Maximizing Jetson performance modes..."
if [ -f /usr/sbin/nvpmodel ]; then
    sudo /usr/sbin/nvpmodel -m 0  # 15W mode
    sudo /usr/bin/jetson_clocks   # Max clocks
    echo "Performance modes activated."
else
    echo "nvpmodel not found. Skipping performance tuning."
fi

# 3. Python environment
echo "Setting up Python virtual environment..."
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate

# Install requirements (excluding heavy ones that should use NVIDIA's wheels)
pip install -r requirements.txt

echo "Setup complete! Remember to activate virtual environment:"
echo "  source venv/bin/activate"
