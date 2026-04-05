// ============================================================
// Edge AI Object Detection System
// Preprocessor Implementation
// ============================================================
#include "preprocessor.h"

#include <opencv2/imgproc.hpp>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace jetson {

Preprocessor::Preprocessor(const PreprocessConfig& config)
    : config_(config) {}

void Preprocessor::process(const cv::Mat& input, 
                           std::vector<float>& output,
                           PreprocessedFrame& frame_out) {
    if (input.empty()) {
        throw std::runtime_error("[Preprocessor] Input frame is empty");
    }
    
    frame_out.original_width = input.cols;
    frame_out.original_height = input.rows;
    frame_out.input_width = config_.input_width;
    frame_out.input_height = config_.input_height;
    
    cv::Mat resized;
    
    if (config_.letterbox) {
        // Letterbox resize (preserves aspect ratio)
        resized = letterboxResize(input, 
                                  frame_out.scale_x, frame_out.scale_y,
                                  frame_out.pad_x, frame_out.pad_y);
    } else {
        // Simple resize
        cv::resize(input, resized, 
                   cv::Size(config_.input_width, config_.input_height));
        frame_out.scale_x = static_cast<float>(input.cols) / config_.input_width;
        frame_out.scale_y = static_cast<float>(input.rows) / config_.input_height;
        frame_out.pad_x = 0.0f;
        frame_out.pad_y = 0.0f;
    }
    
    // Convert to float CHW normalized tensor
    hwcToChw(resized, output);
}

void Preprocessor::processBatch(const std::vector<cv::Mat>& inputs,
                                std::vector<float>& output,
                                std::vector<PreprocessedFrame>& frames_out) {
    frames_out.resize(inputs.size());
    
    size_t single_size = 3 * config_.input_width * config_.input_height;
    output.resize(inputs.size() * single_size);
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<float> single_output;
        process(inputs[i], single_output, frames_out[i]);
        std::memcpy(output.data() + i * single_size, 
                    single_output.data(), 
                    single_size * sizeof(float));
    }
}

void Preprocessor::scaleDetections(std::vector<Detection>& detections,
                                    const PreprocessedFrame& frame_info) const {
    for (auto& det : detections) {
        if (config_.letterbox) {
            // Remove padding offset
            det.bbox.x1 = (det.bbox.x1 - frame_info.pad_x) / frame_info.scale_x;
            det.bbox.y1 = (det.bbox.y1 - frame_info.pad_y) / frame_info.scale_y;
            det.bbox.x2 = (det.bbox.x2 - frame_info.pad_x) / frame_info.scale_x;
            det.bbox.y2 = (det.bbox.y2 - frame_info.pad_y) / frame_info.scale_y;
        } else {
            det.bbox.x1 *= frame_info.scale_x;
            det.bbox.y1 *= frame_info.scale_y;
            det.bbox.x2 *= frame_info.scale_x;
            det.bbox.y2 *= frame_info.scale_y;
        }
        
        // Clamp to image bounds
        det.bbox.x1 = std::max(0.0f, std::min(det.bbox.x1, 
                      static_cast<float>(frame_info.original_width)));
        det.bbox.y1 = std::max(0.0f, std::min(det.bbox.y1, 
                      static_cast<float>(frame_info.original_height)));
        det.bbox.x2 = std::max(0.0f, std::min(det.bbox.x2, 
                      static_cast<float>(frame_info.original_width)));
        det.bbox.y2 = std::max(0.0f, std::min(det.bbox.y2, 
                      static_cast<float>(frame_info.original_height)));
    }
}

cv::Mat Preprocessor::letterboxResize(const cv::Mat& input,
                                       float& scale_x, float& scale_y,
                                       float& pad_x, float& pad_y) {
    int iw = input.cols;
    int ih = input.rows;
    int tw = config_.input_width;
    int th = config_.input_height;
    
    // Calculate scale to fit the image within target dimensions
    float scale = std::min(static_cast<float>(tw) / iw, 
                           static_cast<float>(th) / ih);
    
    int nw = static_cast<int>(iw * scale);
    int nh = static_cast<int>(ih * scale);
    
    // Padding
    pad_x = (tw - nw) / 2.0f;
    pad_y = (th - nh) / 2.0f;
    
    // Scale factors for coordinate mapping
    // Note: these are used to map from model coords back to original coords
    scale_x = scale;
    scale_y = scale;
    
    // Resize
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    
    // Create padded image
    cv::Mat padded(th, tw, CV_8UC3, cv::Scalar(
        static_cast<int>(config_.pad_value),
        static_cast<int>(config_.pad_value),
        static_cast<int>(config_.pad_value)
    ));
    
    // Copy resized image to center
    int top = static_cast<int>(pad_y);
    int left = static_cast<int>(pad_x);
    resized.copyTo(padded(cv::Rect(left, top, nw, nh)));
    
    return padded;
}

void Preprocessor::hwcToChw(const cv::Mat& input, std::vector<float>& output) {
    int channels = input.channels();
    int height = input.rows;
    int width = input.cols;
    
    output.resize(channels * height * width);
    
    // Split channels and convert to float
    std::vector<cv::Mat> channels_mat;
    cv::split(input, channels_mat);
    
    for (int c = 0; c < channels; ++c) {
        cv::Mat float_channel;
        channels_mat[c].convertTo(float_channel, CV_32F);
        
        if (config_.normalize) {
            float_channel /= 255.0f;
        }
        
        // Apply mean/std normalization
        if (config_.mean[c] != 0.0f || config_.std[c] != 1.0f) {
            float_channel = (float_channel - config_.mean[c]) / config_.std[c];
        }
        
        // Copy to CHW layout: channel c starts at offset c * H * W
        size_t channel_offset = c * height * width;
        std::memcpy(output.data() + channel_offset, 
                    float_channel.data, 
                    height * width * sizeof(float));
    }
}

} // namespace jetson
