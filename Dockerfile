# ============================================================
# Edge AI Object Detection System - Dockerfile
# ============================================================
# For Jetson Nano deployment, use:
#   FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime
#
# For development/simulation (x86_64), this uses Ubuntu 22.04
# with CUDA simulation mode enabled.
# ============================================================

# =========================
# Stage 1: Build Environment
# =========================
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    ca-certificates \
    # OpenCV dependencies
    libopencv-dev \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    libopencv-highgui-dev \
    libopencv-videoio-dev \
    libopencv-imgcodecs-dev \
    # GStreamer (for CSI camera simulation)
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY CMakeLists.txt .
COPY cpp/ cpp/
COPY configs/ configs/
COPY tests/ tests/

# Build C++ project in simulation mode
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DSIMULATE_GPU=ON \
        -DUSE_FP16=ON \
        -DBUILD_TESTS=ON && \
    make -j$(nproc)

# =========================
# Stage 2: Python Environment
# =========================
FROM ubuntu:22.04 AS python-env

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy Python source
COPY src/ src/
COPY benchmarks/ benchmarks/
COPY scripts/ scripts/
COPY configs/ configs/

# =========================
# Stage 3: Runtime
# =========================
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-highgui4.5d \
    libopencv-videoio4.5d \
    libopencv-imgcodecs4.5d \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-plugins-good \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built C++ binaries from builder
COPY --from=builder /app/build/jetson_detector /usr/local/bin/
COPY --from=builder /app/build/test_inference /usr/local/bin/
COPY --from=builder /app/build/test_pipeline /usr/local/bin/

# Copy Python environment
COPY --from=python-env /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=python-env /app/src /app/src
COPY --from=python-env /app/benchmarks /app/benchmarks

# Copy configs and scripts
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY models/ /app/models/

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD jetson_detector --health-check || exit 1

# Default entry point
ENTRYPOINT ["jetson_detector"]
CMD ["--source", "synthetic", "--config", "/app/configs/pipeline_config.yaml", "--benchmark"]

# ============================================================
# Usage:
#   docker build -t edge-ai-detector .
#   docker run --rm edge-ai-detector
#   docker run --rm edge-ai-detector --source video --input /app/data/test.mp4
#
# For Jetson Nano with GPU access:
#   docker run --runtime nvidia --rm edge-ai-detector
# ============================================================
