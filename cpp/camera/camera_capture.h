// ============================================================
// Edge AI Object Detection System
// Camera Capture Module
// ============================================================
// Supports multiple video input sources:
// - CSI camera (NVIDIA Jetson GStreamer pipeline)
// - USB webcam
// - Video file
// - Synthetic test frames (no camera required)
// ============================================================
#pragma once

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <atomic>
#include <chrono>

namespace jetson {

/// Camera source type
enum class CameraSource {
    CSI,            // CSI camera via GStreamer (nvarguscamerasrc)
    USB,            // USB webcam
    VIDEO_FILE,     // Pre-recorded video file
    SYNTHETIC       // Generated test frames (no camera needed)
};

/// Camera configuration
struct CameraConfig {
    CameraSource source = CameraSource::SYNTHETIC;
    int width = 1280;
    int height = 720;
    int fps = 30;
    int device_id = 0;               // USB webcam ID
    int csi_sensor_id = 0;           // CSI camera sensor ID
    std::string video_path;           // Path for VIDEO_FILE source
    int flip_method = 0;             // CSI camera flip (0 = none, 2 = 180°)
    
    // Synthetic frame settings
    int synthetic_num_objects = 5;    // Objects per frame
    bool synthetic_motion = true;    // Animate objects between frames
};

/// Camera capture abstraction
class CameraCapture {
public:
    explicit CameraCapture(const CameraConfig& config = {});
    ~CameraCapture();
    
    /// Open the camera/video source
    bool open();
    
    /// Read the next frame
    /// @param frame Output BGR frame
    /// @return true if frame was read successfully
    bool read(cv::Mat& frame);
    
    /// Release the camera
    void release();
    
    /// Check if camera is open
    bool isOpened() const;
    
    // Properties
    int getWidth() const { return config_.width; }
    int getHeight() const { return config_.height; }
    double getFPS() const { return config_.fps; }
    CameraSource getSource() const { return config_.source; }
    std::string getSourceName() const;
    int getFrameCount() const { return frame_count_.load(); }
    
private:
    CameraConfig config_;
    cv::VideoCapture cap_;
    std::atomic<int> frame_count_{0};
    
    // Synthetic frame generation state
    int synthetic_frame_id_ = 0;
    std::chrono::steady_clock::time_point last_frame_time_;
    
    /// Build GStreamer pipeline string for CSI camera
    std::string buildGStreamerPipeline() const;
    
    /// Generate a synthetic test frame
    void generateSyntheticFrame(cv::Mat& frame);
};

/// Parse camera source from string
CameraSource parseCameraSource(const std::string& source_str);

/// Get camera source display name
std::string cameraSourceToString(CameraSource source);

} // namespace jetson
