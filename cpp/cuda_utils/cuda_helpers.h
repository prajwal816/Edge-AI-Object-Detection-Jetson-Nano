// ============================================================
// Edge AI Object Detection System
// CUDA Helper Utilities
// ============================================================
// Provides CUDA abstractions that work in both real CUDA and
// simulation mode. On systems without CUDA, all operations
// are simulated with realistic timing behavior.
// ============================================================
#pragma once

#include <cstddef>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <iomanip>
#include <functional>

namespace jetson {
namespace cuda {

// ============================================================
// Error Handling
// ============================================================

#ifdef SIMULATE_GPU

#define CUDA_CHECK(call) do { /* simulated - always succeeds */ } while(0)
#define CUDA_CHECK_LAST() do { /* simulated */ } while(0)

#else

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::ostringstream oss;                                         \
            oss << "CUDA Error at " << __FILE__ << ":" << __LINE__          \
                << " - " << cudaGetErrorString(err);                        \
            throw std::runtime_error(oss.str());                            \
        }                                                                   \
    } while(0)

#define CUDA_CHECK_LAST()                                                   \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            std::ostringstream oss;                                         \
            oss << "CUDA Error at " << __FILE__ << ":" << __LINE__          \
                << " - " << cudaGetErrorString(err);                        \
            throw std::runtime_error(oss.str());                            \
        }                                                                   \
    } while(0)

#endif

// ============================================================
// CUDA Stream Wrapper
// ============================================================

class CudaStream {
public:
    CudaStream() {
#ifndef SIMULATE_GPU
        CUDA_CHECK(cudaStreamCreate(&stream_));
#endif
        created_ = true;
    }
    
    ~CudaStream() {
        if (created_) {
#ifndef SIMULATE_GPU
            cudaStreamDestroy(stream_);
#endif
        }
    }

    // Non-copyable
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // Movable
    CudaStream(CudaStream&& other) noexcept {
#ifndef SIMULATE_GPU
        stream_ = other.stream_;
#endif
        created_ = other.created_;
        other.created_ = false;
    }

    void synchronize() {
#ifndef SIMULATE_GPU
        CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif
    }

#ifndef SIMULATE_GPU
    cudaStream_t get() const { return stream_; }
#endif

private:
#ifndef SIMULATE_GPU
    cudaStream_t stream_{};
#endif
    bool created_ = false;
};

// ============================================================
// CUDA Event Timer
// ============================================================

class CudaTimer {
public:
    CudaTimer() {
#ifndef SIMULATE_GPU
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
#endif
    }

    ~CudaTimer() {
#ifndef SIMULATE_GPU
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
#endif
    }

    void start() {
#ifdef SIMULATE_GPU
        start_time_ = std::chrono::high_resolution_clock::now();
#else
        CUDA_CHECK(cudaEventRecord(start_));
#endif
    }

    void stop() {
#ifdef SIMULATE_GPU
        stop_time_ = std::chrono::high_resolution_clock::now();
#else
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
#endif
    }

    /// Returns elapsed time in milliseconds
    float elapsedMs() const {
#ifdef SIMULATE_GPU
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            stop_time_ - start_time_);
        return duration.count() / 1000.0f;
#else
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
#endif
    }

private:
#ifdef SIMULATE_GPU
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stop_time_;
#else
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
#endif
};

// ============================================================
// CUDA Memory Allocation Wrappers
// ============================================================

/// Allocate GPU memory (or host memory in simulation)
inline void* cudaMallocWrapper(size_t size) {
    void* ptr = nullptr;
#ifdef SIMULATE_GPU
    ptr = std::malloc(size);
    if (!ptr) {
        throw std::runtime_error("Simulated CUDA malloc failed: " + std::to_string(size) + " bytes");
    }
    std::memset(ptr, 0, size);
#else
    CUDA_CHECK(cudaMalloc(&ptr, size));
#endif
    return ptr;
}

/// Free GPU memory
inline void cudaFreeWrapper(void* ptr) {
    if (!ptr) return;
#ifdef SIMULATE_GPU
    std::free(ptr);
#else
    CUDA_CHECK(cudaFree(ptr));
#endif
}

/// Copy data host → device
inline void cudaMemcpyH2D(void* dst, const void* src, size_t size) {
#ifdef SIMULATE_GPU
    std::memcpy(dst, src, size);
#else
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
#endif
}

/// Copy data device → host
inline void cudaMemcpyD2H(void* dst, const void* src, size_t size) {
#ifdef SIMULATE_GPU
    std::memcpy(dst, src, size);
#else
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
#endif
}

/// Copy data device → device
inline void cudaMemcpyD2D(void* dst, const void* src, size_t size) {
#ifdef SIMULATE_GPU
    std::memcpy(dst, src, size);
#else
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
#endif
}

// ============================================================
// Zero-Copy Buffer (Unified Memory Abstraction)
// ============================================================

/// Zero-copy buffer that can be accessed by both CPU and GPU
/// On Jetson, this uses cudaMallocManaged for true zero-copy
template<typename T>
class ZeroCopyBuffer {
public:
    ZeroCopyBuffer() = default;
    
    explicit ZeroCopyBuffer(size_t count) {
        allocate(count);
    }

    ~ZeroCopyBuffer() {
        deallocate();
    }

    // Non-copyable
    ZeroCopyBuffer(const ZeroCopyBuffer&) = delete;
    ZeroCopyBuffer& operator=(const ZeroCopyBuffer&) = delete;
    
    // Movable
    ZeroCopyBuffer(ZeroCopyBuffer&& other) noexcept 
        : data_(other.data_), count_(other.count_), bytes_(other.bytes_) {
        other.data_ = nullptr;
        other.count_ = 0;
        other.bytes_ = 0;
    }

    ZeroCopyBuffer& operator=(ZeroCopyBuffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            data_ = other.data_;
            count_ = other.count_;
            bytes_ = other.bytes_;
            other.data_ = nullptr;
            other.count_ = 0;
            other.bytes_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        deallocate();
        count_ = count;
        bytes_ = count * sizeof(T);
#ifdef SIMULATE_GPU
        data_ = static_cast<T*>(std::malloc(bytes_));
        if (!data_) throw std::runtime_error("Zero-copy allocation failed");
        std::memset(data_, 0, bytes_);
#else
        CUDA_CHECK(cudaMallocManaged(&data_, bytes_));
#endif
    }

    void deallocate() {
        if (data_) {
#ifdef SIMULATE_GPU
            std::free(data_);
#else
            CUDA_CHECK(cudaFree(data_));
#endif
            data_ = nullptr;
            count_ = 0;
            bytes_ = 0;
        }
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t count() const { return count_; }
    size_t bytes() const { return bytes_; }
    
    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
    size_t bytes_ = 0;
};

// ============================================================
// GPU Device Information
// ============================================================

struct DeviceInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    size_t total_memory_mb = 0;
    size_t free_memory_mb = 0;
    int multiprocessor_count = 0;
    int max_threads_per_block = 0;
    
    std::string toString() const {
        std::ostringstream oss;
        oss << "GPU: " << name 
            << " (SM " << major << "." << minor << ")"
            << " | Memory: " << free_memory_mb << "/" << total_memory_mb << " MB"
            << " | SMs: " << multiprocessor_count;
        return oss.str();
    }
};

inline DeviceInfo getDeviceInfo() {
    DeviceInfo info;
#ifdef SIMULATE_GPU
    // Simulated Jetson Nano specifications
    info.name = "NVIDIA Tegra X1 (Jetson Nano) [SIMULATED]";
    info.major = 5;
    info.minor = 3;
    info.total_memory_mb = 4096;
    info.free_memory_mb = 3200;
    info.multiprocessor_count = 1;
    info.max_threads_per_block = 1024;
#else
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_memory_mb = free_mem / (1024 * 1024);
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
#endif
    return info;
}

} // namespace cuda
} // namespace jetson
