# API Reference

This document provides a high-level overview of the exposed C++ and Python API.

## Python Pipeline API

### `src/python/utils/config.py`
Provides unified YAML-to-Class dataclasses.
- `PipelineConfig`: Master config block.
- `ModelConfig`: Defines model parameters like anchor sizes.
- `build_pipeline_config(yaml_path)`: Factory to generate object tree.

### `src/python/training/train.py`
Click CLI for training YOLOv8 frameworks on Jetson hardware datasets.
- Imports `ultralytics`. Output `best.pt`.
- Automatically calls `dataset_generator.py` if dataset path is empty.

### `src/python/export/tensorrt_build.py`
- `_build_real_engine()`: Translates ONNX networks into executable engine formats. Sets FP16 or INT8 scaling vectors.

## C++ Pipeline API

### `class InferencePipeline` (cpp/pipeline/pipeline.h)
The central manager of the end-to-end multi-threaded system.
```cpp
bool initialize(); // Validates config, starts engine
void start();      // Spawns 4 pipeline threads
void stop();       // Triggers shutdown condition variables
PipelineMetrics getMetrics() const;
```

### `class GPUMemoryPool` (cpp/cuda_utils/gpu_memory_pool.h)
Static GPU arena memory manager for eliminating allocation stalls.
```cpp
void* allocate(size_t size, std::string tag);
template<typename T> T* allocateTyped(size_t count);
void reset(); // Triggered at EOF
```

### `class ThreadSafeQueue<T>` (cpp/pipeline/thread_safe_queue.h)
Condition variable driven concurrent queues handling stage-to-stage transfers.
```cpp
bool push(T item, std::chrono::milliseconds timeout);
std::optional<T> pop(std::chrono::milliseconds timeout);
void shutdown(); // Wakes up wait states
```

### `class TRTEngine` (cpp/inference/trt_engine.h)
A wrapper explicitly designed around `nvinfer1` constructs.
```cpp
bool loadEngine();
std::vector<Detection> infer(const std::vector<float>& frame);
std::vector<std::vector<Detection>> inferBatch(const std::vector<std::vector<float>>& frames);
```
