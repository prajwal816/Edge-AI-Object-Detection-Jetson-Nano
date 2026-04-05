// ============================================================
// Edge AI Object Detection System
// Thread-Safe Bounded Queue
// ============================================================
// Lock-based bounded queue for inter-thread communication
// in the multi-stage inference pipeline. Supports blocking
// push/pop with timeout, and backpressure when full.
// ============================================================
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>
#include <atomic>

namespace jetson {

template<typename T>
class ThreadSafeQueue {
public:
    /// @param max_size Maximum queue capacity (0 = unlimited)
    explicit ThreadSafeQueue(size_t max_size = 16)
        : max_size_(max_size) {}
    
    /// Push an item (blocks if queue is full)
    /// @return false if queue was shut down
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] {
            return shutdown_ || (max_size_ == 0) || (queue_.size() < max_size_);
        });
        
        if (shutdown_) return false;
        
        queue_.push(std::move(item));
        total_pushed_++;
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }
    
    /// Push with timeout
    /// @return false if timed out or queue is shut down
    bool push(T item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool result = not_full_.wait_for(lock, timeout, [this] {
            return shutdown_ || (max_size_ == 0) || (queue_.size() < max_size_);
        });
        
        if (!result || shutdown_) return false;
        
        queue_.push(std::move(item));
        total_pushed_++;
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }
    
    /// Pop an item (blocks if queue is empty)
    /// @return false if queue was shut down and is empty
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] {
            return shutdown_ || !queue_.empty();
        });
        
        if (queue_.empty()) return false;
        
        item = std::move(queue_.front());
        queue_.pop();
        total_popped_++;
        lock.unlock();
        not_full_.notify_one();
        return true;
    }
    
    /// Pop with timeout
    /// @return std::nullopt if timed out or shut down
    std::optional<T> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool result = not_empty_.wait_for(lock, timeout, [this] {
            return shutdown_ || !queue_.empty();
        });
        
        if (!result || queue_.empty()) return std::nullopt;
        
        T item = std::move(queue_.front());
        queue_.pop();
        total_popped_++;
        lock.unlock();
        not_full_.notify_one();
        return item;
    }
    
    /// Try to pop without blocking
    bool tryPop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        
        item = std::move(queue_.front());
        queue_.pop();
        total_popped_++;
        return true;
    }
    
    /// Shut down the queue (unblocks all waiters)
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }
    
    /// Clear all items in the queue
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }
    
    /// Reset for reuse (after shutdown)
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
        shutdown_ = false;
        total_pushed_ = 0;
        total_popped_ = 0;
    }
    
    // Queue state
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    bool isShutdown() const { return shutdown_; }
    size_t totalPushed() const { return total_pushed_; }
    size_t totalPopped() const { return total_popped_; }
    
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    size_t max_size_;
    std::atomic<bool> shutdown_{false};
    std::atomic<size_t> total_pushed_{0};
    std::atomic<size_t> total_popped_{0};
};

} // namespace jetson
