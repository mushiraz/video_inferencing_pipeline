#include "memory_manager.hpp"
#include <iostream>
#include <algorithm>

namespace qwen {

MemoryManager& MemoryManager::instance() {
    static MemoryManager instance;
    return instance;
}

void MemoryManager::track_allocation(const std::string& tag, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    allocations_[tag] += bytes;
    total_allocated_ += bytes;
    
    size_t current_total = total_allocated_.load();
    size_t current_peak = peak_usage_.load();
    
    while (current_total > current_peak && 
           !peak_usage_.compare_exchange_weak(current_peak, current_total)) {
        current_peak = peak_usage_.load();
    }
}

void MemoryManager::track_deallocation(const std::string& tag, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(tag);
    if (it != allocations_.end()) {
        it->second = (it->second >= bytes) ? it->second - bytes : 0;
        if (it->second == 0) {
            allocations_.erase(it);
        }
    }
    
    total_allocated_ = (total_allocated_ >= bytes) ? total_allocated_ - bytes : 0;
}

size_t MemoryManager::get_total_allocated() const {
    return total_allocated_.load();
}

size_t MemoryManager::get_peak_usage() const {
    return peak_usage_.load();
}

std::unordered_map<std::string, size_t> MemoryManager::get_allocation_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_;
}

bool MemoryManager::is_memory_available(size_t required_bytes) const {
    size_t current_usage = total_allocated_.load();
    size_t limit = memory_limit_.load();
    
    return (current_usage + required_bytes) <= limit;
}

void MemoryManager::garbage_collect() {
    // Force garbage collection - in a real implementation, this might
    // trigger cleanup of cached data, temporary files, etc.
    std::cout << "Memory garbage collection triggered. Current usage: " 
              << (get_total_allocated() / (1024 * 1024)) << " MB" << std::endl;
    
    // For now, just print memory statistics
    auto stats = get_allocation_stats();
    for (const auto& [tag, bytes] : stats) {
        std::cout << "  " << tag << ": " << (bytes / (1024 * 1024)) << " MB" << std::endl;
    }
}

void MemoryManager::set_memory_limit(size_t limit_bytes) {
    memory_limit_.store(limit_bytes);
    std::cout << "Memory limit set to " << (limit_bytes / (1024 * 1024)) << " MB" << std::endl;
}

} // namespace qwen 