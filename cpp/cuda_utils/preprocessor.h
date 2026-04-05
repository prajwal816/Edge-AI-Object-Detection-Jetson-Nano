// ============================================================
// Edge AI Object Detection System
// GPU Preprocessor
// ============================================================
// Handles image preprocessing for inference:
// - Resize with letterboxing (aspect ratio preservation)
// - Normalize to [0, 1]
// - HWC → CHW conversion
// Uses OpenCV CUDA when available, falls back to CPU
// ============================================================
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "cuda_helpers.h"
#include "../inference/detection.h"

namespace jetson {

/// Preprocessing configuration
struct PreprocessConfig {
    int input_width = 640;
    int input_height = 640;
    float mean[3] = {0.0f, 0.0f, 0.0f};      // BGR mean subtraction
    float std[3] = {1.0f, 1.0f, 1.0f};        // BGR std division
    bool normalize = true;                      // Scale to [0, 1]
    bool letterbox = true;                      // Maintain aspect ratio
    float pad_value = 114.0f;                   // Padding fill value
};

/// Image preprocessor for TensorRT inference
class Preprocessor {
public:
    explicit Preprocessor(const PreprocessConfig& config = {});
    
    /// Preprocess a single frame (BGR cv::Mat → float CHW tensor)
    /// @param input     Input BGR image
    /// @param output    Output float tensor (pre-allocated or will be resized)
    /// @param frame_out PreprocessedFrame metadata (scale, padding info)
    void process(const cv::Mat& input, 
                 std::vector<float>& output,
                 PreprocessedFrame& frame_out);
    
    /// Preprocess a batch of frames
    void processBatch(const std::vector<cv::Mat>& inputs,
                      std::vector<float>& output,
                      std::vector<PreprocessedFrame>& frames_out);
    
    /// Reverse the coordinate transform (map detection coords back to original image)
    void scaleDetections(std::vector<Detection>& detections,
                         const PreprocessedFrame& frame_info) const;
    
    const PreprocessConfig& getConfig() const { return config_; }
    
private:
    PreprocessConfig config_;
    
    /// Letterbox resize preserving aspect ratio
    cv::Mat letterboxResize(const cv::Mat& input, 
                            float& scale_x, float& scale_y,
                            float& pad_x, float& pad_y);
    
    /// Convert BGR HWC uint8 to CHW float32 normalized
    void hwcToChw(const cv::Mat& input, std::vector<float>& output);
};

} // namespace jetson
