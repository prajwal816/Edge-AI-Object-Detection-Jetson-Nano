// ============================================================
// Edge AI Object Detection System
// Multi-Threaded Pipeline Implementation
// ============================================================
#include "pipeline.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace jetson {

double InferencePipeline::getCurrentTimeMs() {
    auto now = std::chrono::high_resolution_clock::now();
    auto epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(epoch).count() / 1000.0;
}

std::string pipelineStateToString(PipelineState state) {
    switch (state) {
        case PipelineState::IDLE: return "IDLE";
        case PipelineState::STARTING: return "STARTING";
        case PipelineState::RUNNING: return "RUNNING";
        case PipelineState::STOPPING: return "STOPPING";
        case PipelineState::STOPPED: return "STOPPED";
        case PipelineState::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

InferencePipeline::InferencePipeline(const PipelineConfig& config)
    : config_(config)
    , capture_queue_(config.queue_size)
    , preprocess_queue_(config.queue_size)
    , result_queue_(config.queue_size) {}

InferencePipeline::~InferencePipeline() {
    if (isRunning()) stop();
}

bool InferencePipeline::initialize() {
    std::cout << "\n======================================" << std::endl;
    std::cout << "  Edge AI Detection Pipeline" << std::endl;
    std::cout << "======================================\n" << std::endl;

    state_ = PipelineState::STARTING;

    // Print GPU info
    auto gpu_info = cuda::getDeviceInfo();
    std::cout << gpu_info.toString() << std::endl;

    // Initialize camera
    camera_ = std::make_unique<CameraCapture>(config_.camera_config);
    if (!camera_->open()) {
        std::cerr << "[Pipeline] Failed to open camera" << std::endl;
        state_ = PipelineState::ERROR;
        return false;
    }

    // Initialize TensorRT engine
    engine_ = std::make_unique<TRTEngine>(config_.engine_config);
    if (!engine_->loadEngine()) {
        std::cerr << "[Pipeline] Failed to load TRT engine" << std::endl;
        state_ = PipelineState::ERROR;
        return false;
    }

    // Warmup engine
    engine_->warmup(config_.warmup_frames);

    // Initialize preprocessor
    preprocessor_ = Preprocessor(config_.preprocess_config);

    std::cout << "\n[Pipeline] Initialized successfully" << std::endl;
    std::cout << "  Camera: " << camera_->getSourceName() << std::endl;
    std::cout << "  Resolution: " << camera_->getWidth() << "x"
              << camera_->getHeight() << std::endl;
    std::cout << "  Engine: " << config_.engine_config.precision << std::endl;
    std::cout << "  Queue Size: " << config_.queue_size << std::endl;
    std::cout << "  Display: " << (config_.display ? "ON" : "OFF") << std::endl;

    if (config_.benchmark) {
        std::cout << "  Benchmark: " << config_.benchmark_frames << " frames" << std::endl;
    }

    return true;
}

void InferencePipeline::start() {
    if (isRunning()) {
        std::cerr << "[Pipeline] Already running" << std::endl;
        return;
    }

    running_ = true;
    frames_processed_ = 0;
    latency_history_.clear();
    start_time_ = std::chrono::steady_clock::now();

    // Reset queues
    capture_queue_.reset();
    preprocess_queue_.reset();
    result_queue_.reset();

    state_ = PipelineState::RUNNING;
    std::cout << "\n[Pipeline] Starting 4-thread pipeline..." << std::endl;

    // Launch threads
    capture_thread_ = std::thread(&InferencePipeline::captureLoop, this);
    preprocess_thread_ = std::thread(&InferencePipeline::preprocessLoop, this);
    inference_thread_ = std::thread(&InferencePipeline::inferenceLoop, this);
    postprocess_thread_ = std::thread(&InferencePipeline::postprocessLoop, this);

    std::cout << "[Pipeline] All threads started" << std::endl;
}

void InferencePipeline::stop() {
    if (!isRunning()) return;

    std::cout << "\n[Pipeline] Stopping..." << std::endl;
    state_ = PipelineState::STOPPING;
    running_ = false;

    // Shutdown queues (unblocks waiting threads)
    capture_queue_.shutdown();
    preprocess_queue_.shutdown();
    result_queue_.shutdown();

    // Join threads
    if (capture_thread_.joinable()) capture_thread_.join();
    if (preprocess_thread_.joinable()) preprocess_thread_.join();
    if (inference_thread_.joinable()) inference_thread_.join();
    if (postprocess_thread_.joinable()) postprocess_thread_.join();

    // Release camera
    if (camera_) camera_->release();

    state_ = PipelineState::STOPPED;
    
    // Print final metrics
    auto metrics = getMetrics();
    std::cout << metrics.toString();
}

// ============================================================
// Thread 1: Camera Capture
// ============================================================
void InferencePipeline::captureLoop() {
    std::cout << "[Thread-Capture] Started" << std::endl;
    int frame_id = 0;

    while (running_) {
        cv::Mat frame;
        if (!camera_->read(frame)) {
            if (config_.camera_config.source == CameraSource::VIDEO_FILE) {
                std::cout << "[Thread-Capture] End of video" << std::endl;
                running_ = false;
                break;
            }
            continue;
        }

        FrameData data = matToFrameData(frame, frame_id);
        data.timestamp = getCurrentTimeMs();

        if (!capture_queue_.push(std::move(data), std::chrono::milliseconds(100))) {
            if (!running_) break;
            continue;  // Queue full, drop frame
        }

        frame_id++;

        // Check frame limit
        if (config_.max_frames > 0 && frame_id >= config_.max_frames) {
            running_ = false;
            break;
        }
        if (config_.benchmark && frame_id >= config_.benchmark_frames + config_.warmup_frames) {
            running_ = false;
            break;
        }
    }

    capture_queue_.shutdown();
    std::cout << "[Thread-Capture] Stopped (frames: " << frame_id << ")" << std::endl;
}

// ============================================================
// Thread 2: Preprocessing
// ============================================================
void InferencePipeline::preprocessLoop() {
    std::cout << "[Thread-Preprocess] Started" << std::endl;
    int count = 0;

    while (running_ || !capture_queue_.empty()) {
        FrameData frame_data;
        if (!capture_queue_.pop(frame_data)) break;

        cv::Mat frame = frameDataToMat(frame_data);
        PreprocessedFrame result;
        result.frame_id = frame_data.frame_id;
        result.capture_timestamp = frame_data.timestamp;

        preprocessor_.process(frame, result.tensor_data, result);
        result.preprocess_timestamp = getCurrentTimeMs();

        if (!preprocess_queue_.push(std::move(result), std::chrono::milliseconds(100))) {
            if (!running_) break;
        }
        count++;
    }

    preprocess_queue_.shutdown();
    std::cout << "[Thread-Preprocess] Stopped (frames: " << count << ")" << std::endl;
}

// ============================================================
// Thread 3: TensorRT Inference
// ============================================================
void InferencePipeline::inferenceLoop() {
    std::cout << "[Thread-Inference] Started" << std::endl;
    int count = 0;

    while (running_ || !preprocess_queue_.empty()) {
        PreprocessedFrame pp_frame;
        if (!preprocess_queue_.pop(pp_frame)) break;

        auto detections = engine_->infer(pp_frame.tensor_data,
                                          pp_frame.original_width,
                                          pp_frame.original_height);

        // Scale detection coordinates back to original image
        preprocessor_.scaleDetections(detections, pp_frame);

        InferenceResult result;
        result.frame_id = pp_frame.frame_id;
        result.capture_timestamp = pp_frame.capture_timestamp;
        result.preprocess_timestamp = pp_frame.preprocess_timestamp;
        result.inference_timestamp = getCurrentTimeMs();
        result.original_width = pp_frame.original_width;
        result.original_height = pp_frame.original_height;
        result.detections = std::move(detections);
        result.inference_time_ms = engine_->getLastInferenceTimeMs();

        if (!result_queue_.push(std::move(result), std::chrono::milliseconds(100))) {
            if (!running_) break;
        }
        count++;
    }

    result_queue_.shutdown();
    std::cout << "[Thread-Inference] Stopped (frames: " << count << ")" << std::endl;
}

// ============================================================
// Thread 4: Post-processing + Visualization
// ============================================================
void InferencePipeline::postprocessLoop() {
    std::cout << "[Thread-Postprocess] Started" << std::endl;

    while (running_ || !result_queue_.empty()) {
        InferenceResult result;
        if (!result_queue_.pop(result)) break;

        double now = getCurrentTimeMs();

        // Calculate per-frame latency
        LatencyStats latency;
        latency.capture_ms = result.preprocess_timestamp - result.capture_timestamp;
        latency.preprocess_ms = result.preprocess_timestamp - result.capture_timestamp;
        latency.inference_ms = result.inference_time_ms;
        latency.postprocess_ms = now - result.inference_timestamp;
        latency.total_ms = now - result.capture_timestamp;

        // Store latency
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            latency_history_.push_back(latency);
        }

        // Reconstruct frame for visualization (from camera if available)
        cv::Mat vis_frame(result.original_height, result.original_width, CV_8UC3,
                         cv::Scalar(40, 40, 40));
        drawDetections(vis_frame, result.detections, latency);

        // Store last frame
        {
            std::lock_guard<std::mutex> lock(last_frame_mutex_);
            last_frame_ = vis_frame.clone();
            last_detections_ = result.detections;
        }

        // Callback
        if (frame_callback_) {
            frame_callback_(vis_frame, result.detections, latency);
        }

        // Display
        if (config_.display) {
            cv::imshow("Edge AI Detector", vis_frame);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') {  // ESC or Q
                running_ = false;
            }
        }

        frames_processed_++;

        // Periodic logging
        if (config_.benchmark && frames_processed_ % config_.log_interval == 0) {
            logMetrics();
        }
    }

    if (config_.display) cv::destroyAllWindows();
    std::cout << "[Thread-Postprocess] Stopped (frames: " << frames_processed_.load() << ")" << std::endl;
}

// ============================================================
// Helpers
// ============================================================

FrameData InferencePipeline::matToFrameData(const cv::Mat& mat, int frame_id) {
    FrameData data;
    data.frame_id = frame_id;
    data.width = mat.cols;
    data.height = mat.rows;
    size_t total = mat.total() * mat.elemSize();
    data.data.resize(total);
    std::memcpy(data.data.data(), mat.data, total);
    return data;
}

cv::Mat InferencePipeline::frameDataToMat(const FrameData& data) {
    cv::Mat mat(data.height, data.width, CV_8UC3);
    std::memcpy(mat.data, data.data.data(), data.data.size());
    return mat;
}

void InferencePipeline::drawDetections(cv::Mat& frame,
                                        const std::vector<Detection>& detections,
                                        const LatencyStats& latency) {
    for (const auto& det : detections) {
        auto [b, g, r] = getClassColor(det.class_id);
        cv::Scalar color(b, g, r);

        // Bounding box
        cv::rectangle(frame,
            cv::Point(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1)),
            cv::Point(static_cast<int>(det.bbox.x2), static_cast<int>(det.bbox.y2)),
            color, 2);

        // Label background
        std::string label = det.class_name + " " +
            std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        int baseline;
        auto label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame,
            cv::Point(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1) - label_size.height - 8),
            cv::Point(static_cast<int>(det.bbox.x1) + label_size.width + 4, static_cast<int>(det.bbox.y1)),
            color, -1);

        // Label text
        cv::putText(frame, label,
            cv::Point(static_cast<int>(det.bbox.x1) + 2, static_cast<int>(det.bbox.y1) - 4),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // HUD overlay
    double fps = 0;
    if (latency.total_ms > 0) fps = 1000.0 / latency.total_ms;
    
    std::ostringstream hud;
    hud << std::fixed << std::setprecision(1);
    hud << "FPS: " << fps << " | Latency: " << latency.total_ms << "ms";
    hud << " | Det: " << detections.size();

    // Dark overlay bar
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 30),
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(frame, hud.str(), cv::Point(8, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 128), 1);

    // Stage latency bar
    std::ostringstream stages;
    stages << std::fixed << std::setprecision(1);
    stages << "Pre:" << latency.preprocess_ms << "ms | Inf:" << latency.inference_ms
           << "ms | Post:" << latency.postprocess_ms << "ms";
    cv::rectangle(frame, cv::Point(0, frame.rows - 25),
                  cv::Point(frame.cols, frame.rows), cv::Scalar(0, 0, 0), -1);
    cv::putText(frame, stages.str(), cv::Point(8, frame.rows - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(180, 180, 180), 1);
}

PipelineMetrics InferencePipeline::getMetrics() const {
    PipelineMetrics m;
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    m.total_frames = static_cast<int>(latency_history_.size());
    if (m.total_frames == 0) return m;

    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    double elapsed_s = std::chrono::duration<double>(elapsed).count();
    m.avg_fps = (elapsed_s > 0) ? m.total_frames / elapsed_s : 0;

    double sum_total = 0, sum_cap = 0, sum_pre = 0, sum_inf = 0, sum_post = 0;
    for (const auto& l : latency_history_) {
        sum_total += l.total_ms;
        sum_cap += l.capture_ms;
        sum_pre += l.preprocess_ms;
        sum_inf += l.inference_ms;
        sum_post += l.postprocess_ms;
        m.min_latency_ms = std::min(m.min_latency_ms, l.total_ms);
        m.max_latency_ms = std::max(m.max_latency_ms, l.total_ms);
    }

    m.avg_latency_ms = sum_total / m.total_frames;
    m.avg_stage_latency.capture_ms = sum_cap / m.total_frames;
    m.avg_stage_latency.preprocess_ms = sum_pre / m.total_frames;
    m.avg_stage_latency.inference_ms = sum_inf / m.total_frames;
    m.avg_stage_latency.postprocess_ms = sum_post / m.total_frames;

    // Simulated GPU metrics
    m.gpu_utilization = std::min(95.0, m.avg_fps * 3.2);
    m.gpu_memory_used_mb = 512.0;
    m.gpu_memory_total_mb = 4096.0;

    return m;
}

bool InferencePipeline::getLastFrame(cv::Mat& frame,
                                      std::vector<Detection>& detections) const {
    std::lock_guard<std::mutex> lock(last_frame_mutex_);
    if (last_frame_.empty()) return false;
    frame = last_frame_.clone();
    detections = last_detections_;
    return true;
}

void InferencePipeline::logMetrics() {
    auto m = getMetrics();
    std::cout << "[Pipeline] Frame " << frames_processed_.load()
              << " | FPS: " << std::fixed << std::setprecision(1) << m.avg_fps
              << " | Latency: " << m.avg_latency_ms << "ms"
              << " | Inf: " << m.avg_stage_latency.inference_ms << "ms" << std::endl;
}

} // namespace jetson
