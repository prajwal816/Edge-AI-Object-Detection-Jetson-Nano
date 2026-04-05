# Jetson Optimization Strategy

Deploying deep learning to edge devices like the NVIDIA Jetson Nano requires specific techniques to maintain real-time performance (30+ FPS). PyTorch-based pipelines typically fail to hit 15 FPS due to Python interpreting overhead, memory management inefficiencies, and lack of graph acceleration.

This project uses the following strategies.

## TensorRT Precision Scaling (FP16 / INT8)
The Jetson Nano's Tegra X1 GPU supports half-precision FP16 calculations.
- **FP32**: Default PyTorch precision. Requires too much memory bandwidth. 
- **FP16**: Shrinks model memory trace by 50%. ~2-3x speedup on Jetson Maxwell architectures. Virtually no loss in Mean Average Precision (mAP).
- **INT8**: Shrinks trace by 75%. Requires an explicit calibration dataset to calculate layer quantization scales.

The pipeline sets precision dynamically via YAML config:
```yaml
pipeline:
  precision: "fp16" # Configures TensorRT BuilderFlag
```

## Buffer Pre-Allocation & Bump Allocators 
A typical deep learning pipeline does the following every frame:
1. Allocate memory for input tensor.
2. Initialize memory for output tensor.
3. Free memory after the inference loop.

`cudaMalloc` triggers host-to-device kernel stalls. The `GPUMemoryPool` circumvents this:
- Boot sequence pre-allocates 256MB of unified memory.
- During inference loop, requests simply return `current_offset += requested_size`.
- End of loop: `offset = 0`. No `cudaFree` is ever called during running loops.

## Concurrent Processing via Thread Queues
Deep learning is sequential but stages do not need to be blocked by each other.
- While the CPU calculates NMS for Frame 0 (Thread 4)
- The GPU runs Inference for Frame 1 (Thread 3)
- The CPU normalizes Frame 2 (Thread 2)
- The ISP captures Frame 3 (Thread 1)

This concurrency means the effective frame pipeline is limited purely by the slowest link (typically Thread 3 - Inference), rather than the absolute sum of all four operations.

## Jetson Max-N Mode
The script `setup_jetson.sh` ensures the Jetson board isn't power-gated.
```bash
sudo /usr/sbin/nvpmodel -m 0  # Enable 10W/15W power modes
sudo /usr/bin/jetson_clocks   # Sets GPU/CPU frequencies to maximum bounds
```
