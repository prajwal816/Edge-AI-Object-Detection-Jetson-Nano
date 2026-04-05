// ============================================================
// Edge AI Object Detection System
// Integration Tests - Pipeline
// ============================================================
#include <iostream>
#include <thread>
#include <chrono>

#include "../cpp/pipeline/pipeline.h"
#include "../cpp/pipeline/thread_safe_queue.h"

int g_tests_passed = 0;
int g_tests_failed = 0;

#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "[FAIL] " << __FUNCTION__ << " line " << __LINE__ \
                      << ": Expected true, got false." << std::endl; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

#define ASSERT_EQ(expected, actual) \
    do { \
        if ((expected) != (actual)) { \
            std::cerr << "[FAIL] " << __FUNCTION__ << " line " << __LINE__ \
                      << ": Expected " << (expected) << ", got " << (actual) << std::endl; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)


// ---------------------------------------------------------
// Tests
// ---------------------------------------------------------

void testThreadSafeQueue() {
    std::cout << "Running testThreadSafeQueue..." << std::endl;
    jetson::ThreadSafeQueue<int> q(5);
    
    ASSERT_TRUE(q.empty());
    
    // Push up to limit
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(q.push(i, std::chrono::milliseconds(10)));
    }
    
    // Over limit should fail with timeout
    ASSERT_TRUE(!q.push(6, std::chrono::milliseconds(10)));
    
    // Pop half
    int val;
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(q.pop(val));
        ASSERT_EQ(i, val);
    }
    
    // Push more
    ASSERT_TRUE(q.push(10));
    
    // Verify sizes
    ASSERT_EQ(5 + 1, q.totalPushed());
    ASSERT_EQ(3, q.totalPopped());
}

void testPipelineLifecycle() {
    std::cout << "Running testPipelineLifecycle..." << std::endl;
    
    jetson::PipelineConfig config;
    config.camera_config.source = jetson::CameraSource::SYNTHETIC;
    config.engine_config.engine_path = "dummy.engine"; // Uses SIMULATE_GPU
    config.max_frames = 10;
    config.display = false;
    
    jetson::InferencePipeline pipeline(config);
    
    ASSERT_EQ(jetson::PipelineState::IDLE, pipeline.getState());
    
    ASSERT_TRUE(pipeline.initialize());
    
    pipeline.start();
    ASSERT_TRUE(pipeline.isRunning());
    
    // Wait for pipeline to finish 10 frames
    while (pipeline.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    ASSERT_EQ(jetson::PipelineState::STOPPED, pipeline.getState());
    
    auto metrics = pipeline.getMetrics();
    ASSERT_EQ(10, metrics.total_frames);
    ASSERT_TRUE(metrics.avg_fps > 0);
}

int main() {
    std::cout << "\n=== Edge AI Pipeline Tests ===\n" << std::endl;
    
    testThreadSafeQueue();
    testPipelineLifecycle();
    
    std::cout << "\nTest Summary:" << std::endl;
    std::cout << "  Passed: " << g_tests_passed << std::endl;
    std::cout << "  Failed: " << g_tests_failed << std::endl;
    
    return g_tests_failed > 0 ? 1 : 0;
}
