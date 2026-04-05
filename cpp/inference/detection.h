// ============================================================
// Edge AI Object Detection System
// Detection Data Types
// ============================================================
#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace jetson {

/// Bounding box in pixel coordinates (x1, y1, x2, y2 format)
struct BBox {
    float x1 = 0.0f;   // Top-left x
    float y1 = 0.0f;   // Top-left y
    float x2 = 0.0f;   // Bottom-right x
    float y2 = 0.0f;   // Bottom-right y

    float width() const { return std::max(0.0f, x2 - x1); }
    float height() const { return std::max(0.0f, y2 - y1); }
    float area() const { return width() * height(); }
    float centerX() const { return (x1 + x2) / 2.0f; }
    float centerY() const { return (y1 + y2) / 2.0f; }

    /// Compute IoU with another bounding box
    float iou(const BBox& other) const {
        float inter_x1 = std::max(x1, other.x1);
        float inter_y1 = std::max(y1, other.y1);
        float inter_x2 = std::min(x2, other.x2);
        float inter_y2 = std::min(y2, other.y2);

        float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_w * inter_h;

        float union_area = area() + other.area() - inter_area;
        return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
    }
};

/// Single object detection result
struct Detection {
    BBox bbox;
    float confidence = 0.0f;
    int class_id = -1;
    std::string class_name;

    std::string toString() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << class_name << " [" << confidence * 100.0f << "%] "
            << "(" << bbox.x1 << ", " << bbox.y1 << ") - "
            << "(" << bbox.x2 << ", " << bbox.y2 << ")";
        return oss.str();
    }
};

/// Frame data passed through the pipeline
struct FrameData {
    int frame_id = 0;
    double timestamp = 0.0;        // Capture timestamp (ms)
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;     // Raw pixel data (BGR)
    
    bool empty() const { return data.empty(); }
};

/// Preprocessed frame ready for inference
struct PreprocessedFrame {
    int frame_id = 0;
    double capture_timestamp = 0.0;
    double preprocess_timestamp = 0.0;
    int original_width = 0;
    int original_height = 0;
    int input_width = 0;
    int input_height = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    std::vector<float> tensor_data;  // CHW normalized float tensor
};

/// Inference result with detections
struct InferenceResult {
    int frame_id = 0;
    double capture_timestamp = 0.0;
    double preprocess_timestamp = 0.0;
    double inference_timestamp = 0.0;
    int original_width = 0;
    int original_height = 0;
    std::vector<Detection> detections;
    float inference_time_ms = 0.0f;
};

/// Per-frame latency breakdown
struct LatencyStats {
    double capture_ms = 0.0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
};

/// Aggregated pipeline performance metrics
struct PipelineMetrics {
    int total_frames = 0;
    double avg_fps = 0.0;
    double avg_latency_ms = 0.0;
    double min_latency_ms = 1e9;
    double max_latency_ms = 0.0;
    
    LatencyStats avg_stage_latency;
    
    double gpu_utilization = 0.0;      // Simulated
    double gpu_memory_used_mb = 0.0;   // Simulated
    double gpu_memory_total_mb = 0.0;  // Simulated
    
    std::string toString() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "\n╔══════════════════════════════════════════╗\n";
        oss << "║       Pipeline Performance Report        ║\n";
        oss << "╠══════════════════════════════════════════╣\n";
        oss << "║  Total Frames:     " << std::setw(8) << total_frames << "             ║\n";
        oss << "║  Average FPS:      " << std::setw(8) << avg_fps << "             ║\n";
        oss << "║  Avg Latency:      " << std::setw(8) << avg_latency_ms << " ms         ║\n";
        oss << "║  Min Latency:      " << std::setw(8) << min_latency_ms << " ms         ║\n";
        oss << "║  Max Latency:      " << std::setw(8) << max_latency_ms << " ms         ║\n";
        oss << "╠══════════════════════════════════════════╣\n";
        oss << "║  Stage Breakdown (avg):                  ║\n";
        oss << "║    Capture:        " << std::setw(8) << avg_stage_latency.capture_ms << " ms         ║\n";
        oss << "║    Preprocess:     " << std::setw(8) << avg_stage_latency.preprocess_ms << " ms         ║\n";
        oss << "║    Inference:      " << std::setw(8) << avg_stage_latency.inference_ms << " ms         ║\n";
        oss << "║    Postprocess:    " << std::setw(8) << avg_stage_latency.postprocess_ms << " ms         ║\n";
        oss << "╠══════════════════════════════════════════╣\n";
        oss << "║  GPU Utilization:  " << std::setw(8) << gpu_utilization << " %          ║\n";
        oss << "║  GPU Memory:       " << std::setw(5) << gpu_memory_used_mb 
            << "/" << std::setw(5) << gpu_memory_total_mb << " MB       ║\n";
        oss << "╚══════════════════════════════════════════╝\n";
        return oss.str();
    }
};

/// COCO class names (80 classes)
inline const std::vector<std::string>& getCocoClassNames() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    return names;
}

/// Color palette for visualization
inline std::tuple<int, int, int> getClassColor(int class_id) {
    static const std::vector<std::tuple<int, int, int>> colors = {
        {56, 56, 255},    {151, 157, 255},  {31, 112, 255},   {29, 178, 255},
        {49, 210, 207},   {10, 249, 72},    {23, 204, 146},   {134, 219, 61},
        {182, 210, 57},   {218, 194, 24},   {254, 172, 0},    {253, 138, 0},
        {255, 95, 0},     {255, 37, 34},    {241, 0, 73},     {224, 0, 130},
        {188, 24, 196},   {130, 37, 233},   {75, 55, 240},    {48, 80, 245}
    };
    return colors[class_id % colors.size()];
}

/// Apply Non-Maximum Suppression
inline std::vector<Detection> applyNMS(
    std::vector<Detection>& detections,
    float nms_threshold = 0.5f
) {
    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].class_id == detections[j].class_id &&
                detections[i].bbox.iou(detections[j].bbox) > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

} // namespace jetson
