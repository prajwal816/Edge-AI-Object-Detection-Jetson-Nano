// ============================================================
// Edge AI Object Detection System
// TensorRT Inference Engine
// ============================================================
// High-performance inference engine using NVIDIA TensorRT.
// Features:
// - Engine deserialization from .engine file
// - Pre-allocated GPU memory pools
// - CUDA stream async inference
// - Batch inference support
// - Simulation mode for development without GPU
// ============================================================
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>

#include "detection.h"
#include "../cuda_utils/cuda_helpers.h"
#include "../cuda_utils/gpu_memory_pool.h"
#include "../cuda_utils/preprocessor.h"

namespace jetson {

/// Engine configuration
struct EngineConfig {
    std::string engine_path;            // Path to .engine file
    int input_width = 640;
    int input_height = 640;
    int num_classes = 80;
    int max_batch_size = 1;
    float confidence_threshold = 0.45f;
    float nms_threshold = 0.5f;
    std::string precision = "fp16";     // fp32, fp16, int8
    size_t memory_pool_mb = 256;        // GPU memory pool size
    int num_detections = 8400;          // Max detections per image (YOLOv8)
    bool use_cuda_graphs = false;       // CUDA Graph optimization
};

/// TensorRT inference engine
class TRTEngine {
public:
    explicit TRTEngine(const EngineConfig& config);
    ~TRTEngine();
    
    // Non-copyable
    TRTEngine(const TRTEngine&) = delete;
    TRTEngine& operator=(const TRTEngine&) = delete;
    
    /// Load and initialize the engine
    bool loadEngine();
    
    /// Run inference on a single frame
    /// @param frame     Preprocessed float tensor (CHW format)
    /// @param width     Original frame width (for coordinate mapping)
    /// @param height    Original frame height
    /// @return Vector of detections
    std::vector<Detection> infer(const std::vector<float>& frame,
                                  int width = 0, int height = 0);
    
    /// Run batch inference
    std::vector<std::vector<Detection>> inferBatch(
        const std::vector<std::vector<float>>& frames,
        const std::vector<std::pair<int, int>>& sizes);
    
    /// Warm up the engine with dummy inference
    void warmup(int iterations = 10);
    
    /// Get the last inference time in milliseconds
    float getLastInferenceTimeMs() const { return last_inference_ms_.load(); }
    
    /// Get average inference time
    float getAvgInferenceTimeMs() const;
    
    /// Check if engine is loaded and ready
    bool isReady() const { return engine_loaded_; }
    
    /// Get engine info string
    std::string getEngineInfo() const;
    
    const EngineConfig& getConfig() const { return config_; }
    
private:
    EngineConfig config_;
    bool engine_loaded_ = false;
    
    // Memory management
    std::unique_ptr<cuda::GPUMemoryPool> memory_pool_;
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    
    // Preprocessing
    Preprocessor preprocessor_;
    
    // Performance tracking
    std::atomic<float> last_inference_ms_{0.0f};
    std::atomic<int> inference_count_{0};
    std::atomic<float> total_inference_ms_{0.0f};
    
    // CUDA resources
    cuda::CudaStream stream_;
    cuda::CudaTimer timer_;
    
    /// Allocate input/output buffers
    void allocateBuffers();
    
    /// Parse raw network output into detections
    std::vector<Detection> parseDetections(const std::vector<float>& raw_output,
                                            int img_width, int img_height);
    
    /// Simulated inference (realistic timing + dummy detections)
    void simulateInference(const std::vector<float>& input,
                           std::vector<float>& output);
};

} // namespace jetson
