// ============================================================
// Edge AI Object Detection System
// GPU Memory Pool
// ============================================================
// Pre-allocated GPU memory pool with bump allocator for
// zero-fragmentation per-frame allocations. Eliminates
// cudaMalloc/cudaFree overhead during inference.
// ============================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <sstream>
#include <iomanip>

#include "cuda_helpers.h"

namespace jetson {
namespace cuda {

/// Configuration for the memory pool
struct MemoryPoolConfig {
    size_t pool_size_mb = 256;          // Total pool size in MB
    size_t alignment = 256;             // Memory alignment (bytes)
    bool enable_tracking = true;        // Track allocations
    bool zero_on_reset = false;         // Zero memory on reset
};

/// Tracks a single allocation within the pool
struct AllocationRecord {
    void* ptr = nullptr;
    size_t size = 0;
    std::string tag;                    // Debug label
};

/// Pre-allocated GPU memory pool with bump allocator
class GPUMemoryPool {
public:
    explicit GPUMemoryPool(const MemoryPoolConfig& config = {});
    ~GPUMemoryPool();
    
    // Non-copyable, non-movable
    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

    /// Initialize the memory pool (allocates GPU memory)
    bool initialize();
    
    /// Allocate memory from the pool (bump allocator)
    /// @param size      Number of bytes to allocate
    /// @param tag       Debug label for the allocation
    /// @return Pointer to allocated memory
    void* allocate(size_t size, const std::string& tag = "");
    
    /// Typed allocation helper
    template<typename T>
    T* allocateTyped(size_t count, const std::string& tag = "") {
        return static_cast<T*>(allocate(count * sizeof(T), tag));
    }
    
    /// Reset the pool (free all allocations, keep pool memory)
    /// This is called at the end of each frame
    void reset();
    
    /// Release all GPU memory
    void release();
    
    // Memory stats
    size_t getUsedBytes() const { return current_offset_.load(); }
    size_t getTotalBytes() const { return pool_size_; }
    size_t getFreeBytes() const { return pool_size_ - current_offset_.load(); }
    size_t getAllocationCount() const { return allocation_count_.load(); }
    size_t getPeakUsedBytes() const { return peak_usage_.load(); }
    
    double getUsedMB() const { return getUsedBytes() / (1024.0 * 1024.0); }
    double getTotalMB() const { return getTotalBytes() / (1024.0 * 1024.0); }
    double getUtilization() const { 
        return pool_size_ > 0 ? (100.0 * getUsedBytes() / pool_size_) : 0.0; 
    }
    
    bool isInitialized() const { return initialized_; }
    
    /// Get formatted memory report
    std::string getReport() const;
    
private:
    MemoryPoolConfig config_;
    void* pool_base_ = nullptr;        // Base pointer of the pool
    size_t pool_size_ = 0;             // Total pool size in bytes
    std::atomic<size_t> current_offset_{0};   // Current bump pointer offset
    std::atomic<size_t> allocation_count_{0};
    std::atomic<size_t> peak_usage_{0};
    bool initialized_ = false;
    
    mutable std::mutex tracking_mutex_;
    std::vector<AllocationRecord> allocations_;
    
    /// Align offset to the configured alignment boundary
    size_t alignOffset(size_t offset) const;
};

} // namespace cuda
} // namespace jetson
