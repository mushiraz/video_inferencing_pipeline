#pragma once

#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string>
#include <cstdlib>

namespace qwen {

class MemoryManager {
public:
    static MemoryManager& instance();
    
    // Memory tracking
    void track_allocation(const std::string& tag, size_t bytes);
    void track_deallocation(const std::string& tag, size_t bytes);
    
    // Memory statistics
    size_t get_total_allocated() const;
    size_t get_peak_usage() const;
    std::unordered_map<std::string, size_t> get_allocation_stats() const;
    
    // Memory management
    bool is_memory_available(size_t required_bytes) const;
    void garbage_collect();
    void set_memory_limit(size_t limit_bytes);
    
    // Custom allocator for large tensors
    template<typename T>
    class TrackedAllocator {
    public:
        using value_type = T;
        
        TrackedAllocator(const std::string& tag) : tag_(tag) {}
        
        T* allocate(size_t n) {
            size_t bytes = n * sizeof(T);
            MemoryManager::instance().track_allocation(tag_, bytes);
            
            // Try aligned allocation first, fallback to regular malloc
            T* ptr = nullptr;
            #if defined(__APPLE__) || defined(__linux__)
                ptr = static_cast<T*>(std::aligned_alloc(64, bytes));
            #endif
            
            if (!ptr) {
                // Fallback to regular allocation
                ptr = static_cast<T*>(std::malloc(bytes));
            }
            
            return ptr;
        }
        
        void deallocate(T* ptr, size_t n) {
            if (ptr) {
                size_t bytes = n * sizeof(T);
                MemoryManager::instance().track_deallocation(tag_, bytes);
                std::free(ptr);
            }
        }
        
    private:
        std::string tag_;
    };

private:
    MemoryManager() = default;
    
    mutable std::mutex mutex_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_usage_{0};
    std::atomic<size_t> memory_limit_{static_cast<size_t>(8) * 1024 * 1024 * 1024}; // 8GB default
    std::unordered_map<std::string, size_t> allocations_;
};

} // namespace qwen 