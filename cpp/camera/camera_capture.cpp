// ============================================================
// Edge AI Object Detection System
// Camera Capture Implementation
// ============================================================
#include "camera_capture.h"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

namespace jetson {

CameraCapture::CameraCapture(const CameraConfig& config)
    : config_(config) {}

CameraCapture::~CameraCapture() {
    release();
}

bool CameraCapture::open() {
    switch (config_.source) {
        case CameraSource::CSI: {
            std::string pipeline = buildGStreamerPipeline();
            std::cout << "[Camera] Opening CSI camera with GStreamer:\n  " 
                      << pipeline << std::endl;
            cap_.open(pipeline, cv::CAP_GSTREAMER);
            break;
        }
        
        case CameraSource::USB: {
            std::cout << "[Camera] Opening USB camera (device " 
                      << config_.device_id << ")" << std::endl;
            cap_.open(config_.device_id);
            if (cap_.isOpened()) {
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
                cap_.set(cv::CAP_PROP_FPS, config_.fps);
            }
            break;
        }
        
        case CameraSource::VIDEO_FILE: {
            if (config_.video_path.empty()) {
                std::cerr << "[Camera] Video file path not specified" << std::endl;
                return false;
            }
            std::cout << "[Camera] Opening video file: " 
                      << config_.video_path << std::endl;
            cap_.open(config_.video_path);
            break;
        }
        
        case CameraSource::SYNTHETIC: {
            std::cout << "[Camera] Using synthetic frame generator ("
                      << config_.width << "x" << config_.height 
                      << " @ " << config_.fps << " FPS)" << std::endl;
            last_frame_time_ = std::chrono::steady_clock::now();
            return true;  // No VideoCapture needed
        }
    }
    
    if (config_.source != CameraSource::SYNTHETIC && !cap_.isOpened()) {
        std::cerr << "[Camera] Failed to open camera source" << std::endl;
        return false;
    }
    
    // Read actual properties
    if (cap_.isOpened()) {
        config_.width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        config_.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        double actual_fps = cap_.get(cv::CAP_PROP_FPS);
        if (actual_fps > 0) config_.fps = static_cast<int>(actual_fps);
    }
    
    std::cout << "[Camera] Opened: " << config_.width << "x" << config_.height 
              << " @ " << config_.fps << " FPS" << std::endl;
    
    return true;
}

bool CameraCapture::read(cv::Mat& frame) {
    if (config_.source == CameraSource::SYNTHETIC) {
        // Rate-limit to configured FPS
        auto now = std::chrono::steady_clock::now();
        auto frame_duration = std::chrono::microseconds(1000000 / config_.fps);
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_frame_time_);
        
        if (elapsed < frame_duration) {
            std::this_thread::sleep_for(frame_duration - elapsed);
        }
        last_frame_time_ = std::chrono::steady_clock::now();
        
        generateSyntheticFrame(frame);
        frame_count_++;
        return true;
    }
    
    if (!cap_.isOpened()) return false;
    
    bool success = cap_.read(frame);
    if (success && !frame.empty()) {
        frame_count_++;
    }
    return success && !frame.empty();
}

void CameraCapture::release() {
    if (cap_.isOpened()) {
        cap_.release();
    }
    std::cout << "[Camera] Released (total frames: " 
              << frame_count_.load() << ")" << std::endl;
}

bool CameraCapture::isOpened() const {
    return config_.source == CameraSource::SYNTHETIC || cap_.isOpened();
}

std::string CameraCapture::getSourceName() const {
    return cameraSourceToString(config_.source);
}

std::string CameraCapture::buildGStreamerPipeline() const {
    // NVIDIA Jetson CSI camera GStreamer pipeline
    // Uses nvarguscamerasrc for hardware-accelerated capture
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc sensor-id=" << config_.csi_sensor_id
             << " ! video/x-raw(memory:NVMM)"
             << ", width=" << config_.width
             << ", height=" << config_.height
             << ", format=NV12"
             << ", framerate=" << config_.fps << "/1"
             << " ! nvvidconv flip-method=" << config_.flip_method
             << " ! video/x-raw, format=BGRx"
             << " ! videoconvert"
             << " ! video/x-raw, format=BGR"
             << " ! appsink drop=1";
    return pipeline.str();
}

void CameraCapture::generateSyntheticFrame(cv::Mat& frame) {
    // Create a visually interesting test frame
    frame = cv::Mat(config_.height, config_.width, CV_8UC3);
    
    // Background gradient
    for (int y = 0; y < config_.height; ++y) {
        for (int x = 0; x < config_.width; ++x) {
            float fx = static_cast<float>(x) / config_.width;
            float fy = static_cast<float>(y) / config_.height;
            
            // Animated gradient using frame_id for motion
            float phase = synthetic_frame_id_ * 0.02f;
            int b = static_cast<int>(30 + 20 * std::sin(fx * 3.14f + phase));
            int g = static_cast<int>(30 + 15 * std::sin(fy * 3.14f + phase * 0.7f));
            int r = static_cast<int>(25 + 10 * std::sin((fx + fy) * 2.0f + phase * 1.3f));
            
            frame.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(std::clamp(b, 0, 255)),
                static_cast<uint8_t>(std::clamp(g, 0, 255)),
                static_cast<uint8_t>(std::clamp(r, 0, 255))
            );
        }
    }
    
    // Simulated objects (representing detected targets)
    std::mt19937 rng(42 + synthetic_frame_id_ / 10);  // Slowly changing seed
    std::uniform_int_distribution<int> x_dist(50, config_.width - 150);
    std::uniform_int_distribution<int> y_dist(50, config_.height - 150);
    std::uniform_int_distribution<int> size_dist(40, 120);
    
    int num_objects = config_.synthetic_num_objects;
    
    // Class colors: person=red, car=blue, bicycle=green
    std::vector<cv::Scalar> colors = {
        cv::Scalar(60, 60, 220),    // Person (red)
        cv::Scalar(220, 140, 60),   // Car (blue)
        cv::Scalar(60, 200, 60),    // Bicycle (green)
        cv::Scalar(220, 200, 60),   // Motorcycle (cyan)
        cv::Scalar(180, 60, 220),   // Bus (magenta)
    };
    
    std::vector<std::string> labels = {"person", "car", "bicycle", "motorcycle", "bus"};
    
    for (int i = 0; i < num_objects; ++i) {
        int cx = x_dist(rng);
        int cy = y_dist(rng);
        int w = size_dist(rng);
        int h = size_dist(rng);
        
        // Add motion
        if (config_.synthetic_motion) {
            float t = synthetic_frame_id_ * 0.05f;
            cx += static_cast<int>(30.0f * std::sin(t + i * 1.5f));
            cy += static_cast<int>(20.0f * std::cos(t * 0.8f + i * 2.0f));
        }
        
        // Clamp to frame
        cx = std::clamp(cx, w / 2, config_.width - w / 2);
        cy = std::clamp(cy, h / 2, config_.height - h / 2);
        
        int class_idx = i % static_cast<int>(colors.size());
        
        // Draw filled rectangle with slight transparency effect
        cv::Rect obj_rect(cx - w / 2, cy - h / 2, w, h);
        cv::rectangle(frame, obj_rect, colors[class_idx], -1);
        
        // Add some internal detail
        cv::Rect inner(cx - w / 4, cy - h / 4, w / 2, h / 2);
        cv::Scalar darker = colors[class_idx] * 0.6;
        cv::rectangle(frame, inner, darker, -1);
        
        // Draw label
        cv::putText(frame, labels[class_idx], 
                    cv::Point(cx - w / 2, cy - h / 2 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    // Frame counter overlay
    std::string frame_text = "SYNTHETIC FRAME #" + std::to_string(synthetic_frame_id_);
    cv::putText(frame, frame_text, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    
    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::string time_str = std::ctime(&time_t);
    time_str.pop_back();  // Remove newline
    cv::putText(frame, time_str, cv::Point(10, config_.height - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150, 150, 150), 1);
    
    synthetic_frame_id_++;
}

CameraSource parseCameraSource(const std::string& source_str) {
    if (source_str == "csi") return CameraSource::CSI;
    if (source_str == "usb") return CameraSource::USB;
    if (source_str == "video") return CameraSource::VIDEO_FILE;
    if (source_str == "synthetic") return CameraSource::SYNTHETIC;
    
    std::cerr << "[Camera] Unknown source '" << source_str 
              << "', defaulting to synthetic" << std::endl;
    return CameraSource::SYNTHETIC;
}

std::string cameraSourceToString(CameraSource source) {
    switch (source) {
        case CameraSource::CSI: return "CSI Camera (GStreamer)";
        case CameraSource::USB: return "USB Webcam";
        case CameraSource::VIDEO_FILE: return "Video File";
        case CameraSource::SYNTHETIC: return "Synthetic Generator";
        default: return "Unknown";
    }
}

} // namespace jetson
