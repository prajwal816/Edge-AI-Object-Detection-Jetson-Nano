// ============================================================
// Edge AI Object Detection System
// Main Entry Point
// ============================================================
// CLI interface for the Jetson Edge AI detection pipeline.
// Supports multiple run modes: live, benchmark, test.
// ============================================================
#include <iostream>
#include <string>
#include <csignal>
#include <fstream>
#include <sstream>
#include <map>
#include <functional>

#include "pipeline/pipeline.h"
#include "inference/detection.h"
#include "cuda_utils/cuda_helpers.h"

// Global pipeline pointer for signal handling
static jetson::InferencePipeline* g_pipeline = nullptr;

void signalHandler(int signum) {
    std::cout << "\n[Main] Caught signal " << signum << ", shutting down..." << std::endl;
    if (g_pipeline) g_pipeline->stop();
}

void printBanner() {
    std::cout << R"(
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ███████╗██████╗  ██████╗ ███████╗     █████╗ ██╗     ║
    ║     ██╔════╝██╔══██╗██╔════╝ ██╔════╝    ██╔══██╗██║     ║
    ║     █████╗  ██║  ██║██║  ███╗█████╗      ███████║██║     ║
    ║     ██╔══╝  ██║  ██║██║   ██║██╔══╝      ██╔══██║██║     ║
    ║     ███████╗██████╔╝╚██████╔╝███████╗    ██║  ██║██║     ║
    ║     ╚══════╝╚═════╝  ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝   ║
    ║                                                           ║
    ║     Object Detection System - NVIDIA Jetson Nano          ║
    ║     YOLOv8 + TensorRT + CUDA Pipeline                     ║
    ║     Version 1.0.0                                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    )" << std::endl;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --source <type>      Camera source: csi, usb, video, synthetic (default: synthetic)\n"
              << "  --engine <path>      Path to TensorRT engine file (default: models/yolov8_det.engine)\n"
              << "  --precision <type>   Inference precision: fp32, fp16, int8 (default: fp16)\n"
              << "  --input <path>       Input video file (for --source video)\n"
              << "  --width <int>        Frame width (default: 1280)\n"
              << "  --height <int>       Frame height (default: 720)\n"
              << "  --fps <int>          Camera FPS (default: 30)\n"
              << "  --benchmark          Run in benchmark mode\n"
              << "  --frames <int>       Number of benchmark frames (default: 300)\n"
              << "  --no-display         Disable visualization window\n"
              << "  --save-output        Save annotated frames to output/\n"
              << "  --config <path>      Path to YAML config file\n"
              << "  --health-check       Run health check and exit\n"
              << "  --help               Show this help message\n"
              << std::endl;
}

struct CLIArgs {
    std::string source = "synthetic";
    std::string engine_path = "models/yolov8_det.engine";
    std::string precision = "fp16";
    std::string input_path;
    std::string config_path;
    int width = 1280;
    int height = 720;
    int fps = 30;
    int frames = 300;
    bool benchmark = false;
    bool display = true;
    bool save_output = false;
    bool health_check = false;
    bool help = false;
};

CLIArgs parseArgs(int argc, char* argv[]) {
    CLIArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { args.help = true; }
        else if (arg == "--health-check") { args.health_check = true; }
        else if (arg == "--benchmark") { args.benchmark = true; }
        else if (arg == "--no-display") { args.display = false; }
        else if (arg == "--save-output") { args.save_output = true; }
        else if (arg == "--source" && i + 1 < argc) { args.source = argv[++i]; }
        else if (arg == "--engine" && i + 1 < argc) { args.engine_path = argv[++i]; }
        else if (arg == "--precision" && i + 1 < argc) { args.precision = argv[++i]; }
        else if (arg == "--input" && i + 1 < argc) { args.input_path = argv[++i]; }
        else if (arg == "--config" && i + 1 < argc) { args.config_path = argv[++i]; }
        else if (arg == "--width" && i + 1 < argc) { args.width = std::stoi(argv[++i]); }
        else if (arg == "--height" && i + 1 < argc) { args.height = std::stoi(argv[++i]); }
        else if (arg == "--fps" && i + 1 < argc) { args.fps = std::stoi(argv[++i]); }
        else if (arg == "--frames" && i + 1 < argc) { args.frames = std::stoi(argv[++i]); }
        else { std::cerr << "Unknown argument: " << arg << std::endl; }
    }
    return args;
}

int runHealthCheck() {
    std::cout << "[Health] Running system health check..." << std::endl;
    auto info = jetson::cuda::getDeviceInfo();
    std::cout << "[Health] GPU: " << info.toString() << std::endl;
    std::cout << "[Health] Memory: " << info.free_memory_mb << "/" << info.total_memory_mb << " MB" << std::endl;
    std::cout << "[Health] Status: OK" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    printBanner();

    CLIArgs args = parseArgs(argc, argv);
    if (args.help) { printUsage(argv[0]); return 0; }
    if (args.health_check) { return runHealthCheck(); }

    // Setup signal handler
    std::signal(SIGINT, signalHandler);
#ifndef _WIN32
    std::signal(SIGTERM, signalHandler);
#endif

    // Build pipeline config from CLI args
    jetson::PipelineConfig config;

    // Engine config
    config.engine_config.engine_path = args.engine_path;
    config.engine_config.precision = args.precision;
    config.engine_config.input_width = 640;
    config.engine_config.input_height = 640;
    config.engine_config.num_classes = 80;
    config.engine_config.confidence_threshold = 0.45f;
    config.engine_config.nms_threshold = 0.5f;
    config.engine_config.memory_pool_mb = 256;

    // Camera config
    config.camera_config.source = jetson::parseCameraSource(args.source);
    config.camera_config.width = args.width;
    config.camera_config.height = args.height;
    config.camera_config.fps = args.fps;
    if (!args.input_path.empty()) {
        config.camera_config.video_path = args.input_path;
    }

    // Pipeline config
    config.queue_size = 4;
    config.warmup_frames = 10;
    config.display = args.display;
    config.save_output = args.save_output;
    config.benchmark = args.benchmark;
    config.benchmark_frames = args.frames;
    config.log_interval = 50;

    if (args.benchmark) {
        config.max_frames = args.frames + config.warmup_frames;
        std::cout << "[Main] Benchmark mode: " << args.frames << " frames" << std::endl;
    }

    // Create and run pipeline
    jetson::InferencePipeline pipeline(config);
    g_pipeline = &pipeline;

    if (!pipeline.initialize()) {
        std::cerr << "[Main] Pipeline initialization failed" << std::endl;
        return 1;
    }

    pipeline.start();

    // Wait for pipeline to finish (benchmark) or user interrupt
    while (pipeline.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Ensure clean shutdown
    if (pipeline.getState() != jetson::PipelineState::STOPPED) {
        pipeline.stop();
    }

    // Print final metrics
    auto metrics = pipeline.getMetrics();
    
    // Export benchmark results if in benchmark mode
    if (args.benchmark) {
        std::cout << "\n[Main] Exporting benchmark results..." << std::endl;
        
        std::ofstream json_out("benchmarks/results/cpp_benchmark.json");
        if (json_out.is_open()) {
            json_out << "{\n"
                     << "  \"pipeline\": \"C++ TensorRT\",\n"
                     << "  \"precision\": \"" << args.precision << "\",\n"
                     << "  \"total_frames\": " << metrics.total_frames << ",\n"
                     << "  \"avg_fps\": " << std::fixed << std::setprecision(2) << metrics.avg_fps << ",\n"
                     << "  \"avg_latency_ms\": " << metrics.avg_latency_ms << ",\n"
                     << "  \"min_latency_ms\": " << metrics.min_latency_ms << ",\n"
                     << "  \"max_latency_ms\": " << metrics.max_latency_ms << ",\n"
                     << "  \"stage_latency\": {\n"
                     << "    \"capture_ms\": " << metrics.avg_stage_latency.capture_ms << ",\n"
                     << "    \"preprocess_ms\": " << metrics.avg_stage_latency.preprocess_ms << ",\n"
                     << "    \"inference_ms\": " << metrics.avg_stage_latency.inference_ms << ",\n"
                     << "    \"postprocess_ms\": " << metrics.avg_stage_latency.postprocess_ms << "\n"
                     << "  },\n"
                     << "  \"gpu_utilization\": " << metrics.gpu_utilization << ",\n"
                     << "  \"gpu_memory_used_mb\": " << metrics.gpu_memory_used_mb << ",\n"
                     << "  \"gpu_memory_total_mb\": " << metrics.gpu_memory_total_mb << "\n"
                     << "}" << std::endl;
            json_out.close();
            std::cout << "[Main] Results saved to benchmarks/results/cpp_benchmark.json" << std::endl;
        }
    }

    g_pipeline = nullptr;
    std::cout << "\n[Main] Exited cleanly" << std::endl;
    return 0;
}
