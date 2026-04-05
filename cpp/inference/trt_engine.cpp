// ============================================================
// Edge AI Object Detection System
// TensorRT Inference Engine Implementation
// ============================================================
#include "trt_engine.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <thread>
#include <chrono>
#include <cstring>

namespace jetson {

TRTEngine::TRTEngine(const EngineConfig& config)
    : config_(config)
    , preprocessor_(PreprocessConfig{config.input_width, config.input_height,
        {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, true, true, 114.0f}) {}

TRTEngine::~TRTEngine() {
    if (memory_pool_) memory_pool_->release();
    std::cout << "[TRTEngine] Destroyed (inferences: " << inference_count_.load()
              << ", avg: " << getAvgInferenceTimeMs() << " ms)" << std::endl;
}

bool TRTEngine::loadEngine() {
    if (engine_loaded_) return true;
    
    std::cout << "[TRTEngine] Loading engine...\n"
              << "  Path: " << config_.engine_path << "\n"
              << "  Input: " << config_.input_width << "x" << config_.input_height << "\n"
              << "  Classes: " << config_.num_classes << "\n"
              << "  Precision: " << config_.precision << "\n"
              << "  Max Batch: " << config_.max_batch_size << std::endl;

#ifdef SIMULATE_GPU
    std::cout << "  Mode: SIMULATED (no GPU required)" << std::endl;
    std::ifstream file(config_.engine_path, std::ios::binary);
    if (!file.good())
        std::cout << "  Engine file not found, using virtual engine" << std::endl;
    else {
        file.seekg(0, std::ios::end);
        std::cout << "  Engine file size: " << (file.tellg() / 1024) << " KB" << std::endl;
    }
#else
    std::ifstream file(config_.engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "[TRTEngine] Cannot open: " << config_.engine_path << std::endl;
        return false;
    }
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    std::cout << "  Engine size: " << (size / (1024*1024)) << " MB" << std::endl;
#endif
    
    cuda::MemoryPoolConfig pool_config;
    pool_config.pool_size_mb = config_.memory_pool_mb;
    pool_config.alignment = 256;
    memory_pool_ = std::make_unique<cuda::GPUMemoryPool>(pool_config);
    if (!memory_pool_->initialize()) return false;
    
    allocateBuffers();
    engine_loaded_ = true;
    std::cout << "[TRTEngine] Engine loaded successfully" << std::endl;
    std::cout << memory_pool_->getReport();
    return true;
}

void TRTEngine::allocateBuffers() {
    size_t in_size = config_.max_batch_size * 3 * config_.input_width * config_.input_height;
    input_buffer_.resize(in_size, 0.0f);
    size_t out_size = config_.max_batch_size * (4 + config_.num_classes) * config_.num_detections;
    output_buffer_.resize(out_size, 0.0f);
    std::cout << "[TRTEngine] Buffers: input=" << in_size*4/1024 << " KB, output="
              << out_size*4/1024 << " KB" << std::endl;
}

std::vector<Detection> TRTEngine::infer(const std::vector<float>& frame,
                                         int width, int height) {
    if (!engine_loaded_) throw std::runtime_error("[TRTEngine] Not loaded");
    timer_.start();
    
    size_t n = 3 * config_.input_width * config_.input_height;
    if (frame.size() >= n)
        std::memcpy(input_buffer_.data(), frame.data(), n * sizeof(float));

#ifdef SIMULATE_GPU
    simulateInference(input_buffer_, output_buffer_);
#endif
    
    timer_.stop();
    float elapsed = timer_.elapsedMs();
    last_inference_ms_ = elapsed;
    total_inference_ms_ = total_inference_ms_.load() + elapsed;
    inference_count_++;
    
    int iw = (width > 0) ? width : config_.input_width;
    int ih = (height > 0) ? height : config_.input_height;
    return parseDetections(output_buffer_, iw, ih);
}

std::vector<std::vector<Detection>> TRTEngine::inferBatch(
    const std::vector<std::vector<float>>& frames,
    const std::vector<std::pair<int, int>>& sizes) {
    std::vector<std::vector<Detection>> results;
    for (size_t i = 0; i < frames.size(); ++i) {
        int w = (i < sizes.size()) ? sizes[i].first : config_.input_width;
        int h = (i < sizes.size()) ? sizes[i].second : config_.input_height;
        results.push_back(infer(frames[i], w, h));
    }
    return results;
}

void TRTEngine::warmup(int iterations) {
    std::cout << "[TRTEngine] Warming up (" << iterations << " iterations)..." << std::endl;
    std::vector<float> dummy(3 * config_.input_width * config_.input_height, 0.5f);
    for (int i = 0; i < iterations; ++i) infer(dummy);
    inference_count_ = 0;
    total_inference_ms_ = 0.0f;
    std::cout << "[TRTEngine] Warmup complete" << std::endl;
}

float TRTEngine::getAvgInferenceTimeMs() const {
    int c = inference_count_.load();
    return (c > 0) ? total_inference_ms_.load() / c : 0.0f;
}

std::string TRTEngine::getEngineInfo() const {
    std::ostringstream oss;
    oss << "TensorRT Engine:\n  Input: " << config_.input_width << "x"
        << config_.input_height << " | Classes: " << config_.num_classes
        << " | Precision: " << config_.precision
        << " | Inferences: " << inference_count_.load()
        << " | Avg: " << getAvgInferenceTimeMs() << " ms";
    return oss.str();
}

std::vector<Detection> TRTEngine::parseDetections(
    const std::vector<float>& raw, int img_w, int img_h) {
    std::vector<Detection> dets;
    const auto& names = getCocoClassNames();
    int nd = config_.num_detections;
    
    for (int i = 0; i < nd; ++i) {
        float max_conf = 0.0f; int best = -1;
        for (int c = 0; c < config_.num_classes; ++c) {
            float v = raw[(4 + c) * nd + i];
            if (v > max_conf) { max_conf = v; best = c; }
        }
        if (max_conf < config_.confidence_threshold) continue;
        
        float cx = raw[0*nd+i], cy = raw[1*nd+i], w = raw[2*nd+i], h = raw[3*nd+i];
        Detection d;
        d.bbox = {cx-w/2, cy-h/2, cx+w/2, cy+h/2};
        d.confidence = max_conf;
        d.class_id = best;
        d.class_name = (best < (int)names.size()) ? names[best] : "class_" + std::to_string(best);
        dets.push_back(d);
    }
    return applyNMS(dets, config_.nms_threshold);
}

void TRTEngine::simulateInference(const std::vector<float>&,
                                   std::vector<float>& output) {
    float base_ms = 28.0f;
    if (config_.precision == "fp32") base_ms = 55.0f;
    else if (config_.precision == "int8") base_ms = 18.0f;
    
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> noise(0.0f, 1.5f);
    float latency = std::max(base_ms * 0.8f, base_ms + noise(rng));
    std::this_thread::sleep_for(std::chrono::microseconds(int(latency * 1000)));
    
    std::fill(output.begin(), output.end(), 0.0f);
    int nd = config_.num_detections;
    int nf = 3 + (rng() % 5);
    
    std::uniform_real_distribution<float> pos(0.1f, 0.9f);
    std::uniform_real_distribution<float> sz(0.05f, 0.2f);
    std::uniform_real_distribution<float> conf(0.55f, 0.98f);
    std::uniform_int_distribution<int> cls(0, std::min(config_.num_classes-1, 4));
    
    for (int i = 0; i < nf && i < nd; ++i) {
        output[0*nd+i] = pos(rng) * config_.input_width;
        output[1*nd+i] = pos(rng) * config_.input_height;
        output[2*nd+i] = sz(rng) * config_.input_width;
        output[3*nd+i] = sz(rng) * config_.input_height;
        output[(4+cls(rng))*nd+i] = conf(rng);
    }
}

} // namespace jetson
