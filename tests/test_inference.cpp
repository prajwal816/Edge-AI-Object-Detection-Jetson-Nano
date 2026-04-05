// ============================================================
// Edge AI Object Detection System
// Unit Tests - Inference Module
// ============================================================
#include <iostream>
#include <vector>
#include <cmath>

#include "../cpp/inference/trt_engine.h"
#include "../cpp/cuda_utils/gpu_memory_pool.h"

// Simple test framework
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

void testMemoryPool() {
    std::cout << "Running testMemoryPool..." << std::endl;
    jetson::cuda::MemoryPoolConfig config;
    config.pool_size_mb = 16;
    
    jetson::cuda::GPUMemoryPool pool(config);
    ASSERT_TRUE(pool.initialize());
    
    void* ptr1 = pool.allocate(1024, "test1");
    ASSERT_TRUE(ptr1 != nullptr);
    
    void* ptr2 = pool.allocate(2048, "test2");
    ASSERT_TRUE(ptr2 != nullptr);
    
    ASSERT_TRUE(pool.getUsedBytes() >= 1024 + 2048);
    ASSERT_EQ(2, pool.getAllocationCount());
    
    pool.reset();
    ASSERT_EQ(0, pool.getUsedBytes());
    ASSERT_EQ(0, pool.getAllocationCount());
}

void testNMS() {
    std::cout << "Running testNMS..." << std::endl;
    std::vector<jetson::Detection> detections;
    
    // Two highly overlapping boxes
    jetson::Detection d1;
    d1.bbox = {100, 100, 200, 200};
    d1.confidence = 0.9f;
    d1.class_id = 0;
    
    jetson::Detection d2;
    d2.bbox = {105, 105, 195, 195};
    d2.confidence = 0.8f;
    d2.class_id = 0;
    
    // One separate box
    jetson::Detection d3;
    d3.bbox = {300, 300, 400, 400};
    d3.confidence = 0.85f;
    d3.class_id = 0;
    
    detections.push_back(d1);
    detections.push_back(d2);
    detections.push_back(d3);
    
    auto result = jetson::applyNMS(detections, 0.5f);
    
    // d2 should be suppressed by d1
    ASSERT_EQ(2, result.size());
    ASSERT_EQ(0.9f, result[0].confidence); // d1 (sorted by conf)
    ASSERT_EQ(0.85f, result[1].confidence); // d3
}

void testEngineInitialization() {
    std::cout << "Running testEngineInitialization..." << std::endl;
    jetson::EngineConfig config;
    config.engine_path = "models/yolov8_det.engine";
    
    jetson::TRTEngine engine(config);
    ASSERT_TRUE(engine.loadEngine());
    ASSERT_TRUE(engine.isReady());
}

int main() {
    std::cout << "\n=== Edge AI Inference Tests ===\n" << std::endl;
    
    testMemoryPool();
    testNMS();
    testEngineInitialization();
    
    std::cout << "\nTest Summary:" << std::endl;
    std::cout << "  Passed: " << g_tests_passed << std::endl;
    std::cout << "  Failed: " << g_tests_failed << std::endl;
    
    return g_tests_failed > 0 ? 1 : 0;
}
