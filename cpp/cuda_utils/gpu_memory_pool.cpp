// ============================================================
// Edge AI Object Detection System
// GPU Memory Pool Implementation
// ============================================================
#include "gpu_memory_pool.h"

#include <iostream>
#include <algorithm>
#include <cstring>

namespace jetson {
namespace cuda {

GPUMemoryPool::GPUMemoryPool(const MemoryPoolConfig& config)
    : config_(config) {}

GPUMemoryPool::~GPUMemoryPool() {
    release();
}

bool GPUMemoryPool::initialize() {
    if (initialized_) {
        std::cerr << "[MemoryPool] Already initialized" << std::endl;
        return true;
    }
    
    pool_size_ = config_.pool_size_mb * 1024 * 1024;
    
    try {
        pool_base_ = cudaMallocWrapper(pool_size_);
    } catch (const std::exception& e) {
        std::cerr << "[MemoryPool] Failed to allocate " 
                  << config_.pool_size_mb << " MB: " << e.what() << std::endl;
        return false;
    }
    
    current_offset_ = 0;
    allocation_count_ = 0;
    peak_usage_ = 0;
    initialized_ = true;
    
    std::cout << "[MemoryPool] Initialized: " << config_.pool_size_mb 
              << " MB (alignment: " << config_.alignment << " bytes)" << std::endl;
    
    return true;
}

void* GPUMemoryPool::allocate(size_t size, const std::string& tag) {
    if (!initialized_) {
        throw std::runtime_error("[MemoryPool] Not initialized");
    }
    
    if (size == 0) {
        throw std::invalid_argument("[MemoryPool] Cannot allocate 0 bytes");
    }
    
    // Align the current offset
    size_t aligned_offset = alignOffset(current_offset_.load());
    size_t new_offset = aligned_offset + size;
    
    if (new_offset > pool_size_) {
        std::ostringstream oss;
        oss << "[MemoryPool] Out of memory: requested " << size 
            << " bytes, available " << (pool_size_ - aligned_offset) 
            << " bytes (total: " << pool_size_ << ")";
        throw std::runtime_error(oss.str());
    }
    
    void* ptr = static_cast<uint8_t*>(pool_base_) + aligned_offset;
    current_offset_ = new_offset;
    allocation_count_++;
    
    // Track peak usage
    size_t current = new_offset;
    size_t peak = peak_usage_.load();
    while (current > peak && !peak_usage_.compare_exchange_weak(peak, current)) {}
    
    // Record allocation for debugging
    if (config_.enable_tracking) {
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        allocations_.push_back({ptr, size, tag});
    }
    
    return ptr;
}

void GPUMemoryPool::reset() {
    if (!initialized_) return;
    
    if (config_.zero_on_reset) {
        std::memset(pool_base_, 0, current_offset_.load());
    }
    
    current_offset_ = 0;
    allocation_count_ = 0;
    
    if (config_.enable_tracking) {
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        allocations_.clear();
    }
}

void GPUMemoryPool::release() {
    if (!initialized_) return;
    
    if (pool_base_) {
        cudaFreeWrapper(pool_base_);
        pool_base_ = nullptr;
    }
    
    pool_size_ = 0;
    current_offset_ = 0;
    allocation_count_ = 0;
    peak_usage_ = 0;
    initialized_ = false;
    
    {
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        allocations_.clear();
    }
    
    std::cout << "[MemoryPool] Released" << std::endl;
}

std::string GPUMemoryPool::getReport() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "\n┌─────────────────────────────────────┐\n";
    oss << "│       GPU Memory Pool Report        │\n";
    oss << "├─────────────────────────────────────┤\n";
    oss << "│  Total:        " << std::setw(8) << getTotalMB() << " MB        │\n";
    oss << "│  Used:         " << std::setw(8) << getUsedMB() << " MB        │\n";
    oss << "│  Free:         " << std::setw(8) << (getTotalMB() - getUsedMB()) << " MB        │\n";
    oss << "│  Peak:         " << std::setw(8) << (getPeakUsedBytes() / (1024.0 * 1024.0)) << " MB        │\n";
    oss << "│  Utilization:  " << std::setw(8) << getUtilization() << " %         │\n";
    oss << "│  Allocations:  " << std::setw(8) << getAllocationCount() << "            │\n";
    oss << "└─────────────────────────────────────┘\n";
    
    // List tracked allocations
    if (config_.enable_tracking) {
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        if (!allocations_.empty()) {
            oss << "  Allocations:\n";
            for (const auto& alloc : allocations_) {
                oss << "    [" << alloc.tag << "] " 
                    << alloc.size << " bytes (" 
                    << (alloc.size / 1024.0) << " KB)\n";
            }
        }
    }
    
    return oss.str();
}

size_t GPUMemoryPool::alignOffset(size_t offset) const {
    if (config_.alignment <= 1) return offset;
    return (offset + config_.alignment - 1) & ~(config_.alignment - 1);
}

} // namespace cuda
} // namespace jetson
