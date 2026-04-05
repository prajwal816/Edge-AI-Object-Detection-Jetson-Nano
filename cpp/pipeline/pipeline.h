// ============================================================
// Edge AI Object Detection System
// Multi-Threaded Inference Pipeline
// ============================================================
// 4-stage pipeline with dedicated threads:
//   Stage 1: Camera Capture
//   Stage 2: Image Preprocessing  
//   Stage 3: TensorRT Inference
//   Stage 4: Post-processing + Visualization
//
// Thread-safe queues between stages provide buffering
// and backpressure for stable throughput.
// ============================================================
#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <vector>
#include <string>
#include <functional>

#include "thread_safe_queue.h"
#include "../inference/trt_engine.h"
#include "../inference/detection.h"
#include "../camera/camera_capture.h"
#include "../cuda_utils/preprocessor.h"

namespace jetson {

/// Pipeline configuration
struct PipelineConfig {
    // Engine settings
    EngineConfig engine_config;
    
    // Camera settings  
    CameraConfig camera_config;
    
    // Preprocessing settings
    PreprocessConfig preprocess_config;
    
    // Pipeline settings
    size_t queue_size = 4;              // Max items in each inter-thread queue
    int warmup_frames = 10;             // Warmup frames before benchmarking
    int max_frames = 0;                 // 0 = unlimited
    
    // Display settings
    bool display = true;                // Show visualization window
    bool save_output = false;           // Save annotated frames
    std::string output_path = "output"; // Output directory
    
    // Benchmark settings
    bool benchmark = false;             // Enable benchmarking
    int benchmark_frames = 300;         // Number of frames to benchmark
    int log_interval = 50;              // Log metrics every N frames
};

/// Pipeline state
enum class PipelineState {
    IDLE,
    STARTING,
    RUNNING,
    STOPPING,
    STOPPED,
    ERROR
};

/// Callback for post-processed frames
using FrameCallback = std::function<void(const cv::Mat& frame, 
                                          const std::vector<Detection>& detections,
                                          const LatencyStats& latency)>;

/// Multi-threaded inference pipeline
class InferencePipeline {
public:
    explicit InferencePipeline(const PipelineConfig& config);
    ~InferencePipeline();
    
    // Non-copyable
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;
    
    /// Initialize all components (engine, camera, queues)
    bool initialize();
    
    /// Start the pipeline (spawns all threads)
    void start();
    
    /// Stop the pipeline (joins all threads)
    void stop();
    
    /// Check if pipeline is running
    bool isRunning() const { return state_ == PipelineState::RUNNING; }
    
    /// Get current pipeline state
    PipelineState getState() const { return state_.load(); }
    
    /// Get aggregated performance metrics
    PipelineMetrics getMetrics() const;
    
    /// Set callback for processed frames
    void setFrameCallback(FrameCallback callback) { frame_callback_ = std::move(callback); }
    
    /// Get the last processed frame with detections
    bool getLastFrame(cv::Mat& frame, std::vector<Detection>& detections) const;
    
private:
    PipelineConfig config_;
    std::atomic<PipelineState> state_{PipelineState::IDLE};
    
    // Pipeline components
    std::unique_ptr<CameraCapture> camera_;
    std::unique_ptr<TRTEngine> engine_;
    Preprocessor preprocessor_;
    
    // Inter-thread queues
    ThreadSafeQueue<FrameData> capture_queue_;
    ThreadSafeQueue<PreprocessedFrame> preprocess_queue_;
    ThreadSafeQueue<InferenceResult> result_queue_;
    
    // Worker threads
    std::thread capture_thread_;
    std::thread preprocess_thread_;
    std::thread inference_thread_;
    std::thread postprocess_thread_;
    
    // Synchronization
    std::atomic<bool> running_{false};
    std::atomic<int> frames_processed_{0};
    
    // Metrics collection
    mutable std::mutex metrics_mutex_;
    std::vector<LatencyStats> latency_history_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Last frame state
    mutable std::mutex last_frame_mutex_;
    cv::Mat last_frame_;
    std::vector<Detection> last_detections_;
    
    // Frame callback
    FrameCallback frame_callback_;
    
    // === Thread Functions ===
    
    /// Thread 1: Captures frames from camera
    void captureLoop();
    
    /// Thread 2: Preprocesses frames (resize, normalize, HWC→CHW)
    void preprocessLoop();
    
    /// Thread 3: Runs TensorRT inference
    void inferenceLoop();
    
    /// Thread 4: Post-processes results and visualization
    void postprocessLoop();
    
    // === Helpers ===
    
    /// Convert cv::Mat to FrameData
    FrameData matToFrameData(const cv::Mat& mat, int frame_id);
    
    /// Convert FrameData to cv::Mat
    cv::Mat frameDataToMat(const FrameData& data);
    
    /// Draw detections on frame
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections,
                        const LatencyStats& latency);
    
    /// Log pipeline metrics
    void logMetrics();
    
    /// Get current timestamp in milliseconds
    static double getCurrentTimeMs();
};

/// Parse string to PipelineState name
std::string pipelineStateToString(PipelineState state);

} // namespace jetson
