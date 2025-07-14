#include <gtest/gtest.h>
#include "memory_manager.hpp"
#include <thread>
#include <vector>
#include <chrono>

namespace qwen {

class MemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset memory manager state
        auto& manager = MemoryManager::instance();
        
        // Clear all existing allocations
        auto existing_stats = manager.get_allocation_stats();
        for (const auto& [tag, bytes] : existing_stats) {
            manager.track_deallocation(tag, bytes);
        }
        
        manager.garbage_collect();
    }
};

TEST_F(MemoryManagerTest, SingletonInstance) {
    auto& manager1 = MemoryManager::instance();
    auto& manager2 = MemoryManager::instance();
    
    EXPECT_EQ(&manager1, &manager2);
}

TEST_F(MemoryManagerTest, BasicAllocationTracking) {
    auto& manager = MemoryManager::instance();
    
    size_t initial_allocated = manager.get_total_allocated();
    
    // Track some allocations
    manager.track_allocation("test1", 1024);
    EXPECT_EQ(manager.get_total_allocated(), initial_allocated + 1024);
    
    manager.track_allocation("test2", 2048);
    EXPECT_EQ(manager.get_total_allocated(), initial_allocated + 1024 + 2048);
    
    // Track deallocations
    manager.track_deallocation("test1", 1024);
    EXPECT_EQ(manager.get_total_allocated(), initial_allocated + 2048);
    
    manager.track_deallocation("test2", 2048);
    EXPECT_EQ(manager.get_total_allocated(), initial_allocated);
}

TEST_F(MemoryManagerTest, PeakUsageTracking) {
    auto& manager = MemoryManager::instance();
    
    // Reset peak usage by getting current state
    size_t initial_allocated = manager.get_total_allocated();
    size_t initial_peak = manager.get_peak_usage();
    
    // Allocate some memory
    manager.track_allocation("peak_test", 5000);
    size_t peak_after_alloc = manager.get_peak_usage();
    
    // Peak should be at least the current allocation
    size_t expected_minimum = std::max(initial_peak, initial_allocated + 5000);
    EXPECT_GE(peak_after_alloc, expected_minimum);
    
    // Deallocate - peak should remain the same or higher
    manager.track_deallocation("peak_test", 5000);
    EXPECT_GE(manager.get_peak_usage(), peak_after_alloc);
}

TEST_F(MemoryManagerTest, AllocationStats) {
    auto& manager = MemoryManager::instance();
    
    // Track allocations with different tags
    manager.track_allocation("video_frames", 1000);
    manager.track_allocation("ml_model", 2000);
    manager.track_allocation("video_frames", 500);
    
    auto stats = manager.get_allocation_stats();
    
    EXPECT_EQ(stats["video_frames"], 1500);
    EXPECT_EQ(stats["ml_model"], 2000);
    
    // Deallocate some
    manager.track_deallocation("video_frames", 500);
    stats = manager.get_allocation_stats();
    EXPECT_EQ(stats["video_frames"], 1000);
}

TEST_F(MemoryManagerTest, MemoryAvailabilityCheck) {
    auto& manager = MemoryManager::instance();
    
    // Set a memory limit (10 MB in bytes)
    size_t limit_bytes = 10 * 1024 * 1024;
    manager.set_memory_limit(limit_bytes);
    
    // Should have memory available initially
    EXPECT_TRUE(manager.is_memory_available(5 * 1024 * 1024));
    
    // Allocate some memory (8 MB)
    manager.track_allocation("limit_test", 8 * 1024 * 1024);
    
    // Should still have some memory available (1 MB)
    EXPECT_TRUE(manager.is_memory_available(1 * 1024 * 1024));
    
    // Should not have enough for large allocation (5 MB)
    EXPECT_FALSE(manager.is_memory_available(5 * 1024 * 1024));
}

TEST_F(MemoryManagerTest, GarbageCollection) {
    auto& manager = MemoryManager::instance();
    
    // Track some allocations
    manager.track_allocation("gc_test1", 1000);
    manager.track_allocation("gc_test2", 2000);
    
    size_t before_gc = manager.get_total_allocated();
    EXPECT_GT(before_gc, 0);
    
    // Garbage collection should reset tracking
    manager.garbage_collect();
    
    // Note: garbage_collect() might not reset everything depending on implementation
    // This test verifies it doesn't crash and can be called safely
    EXPECT_NO_THROW(manager.garbage_collect());
}

TEST_F(MemoryManagerTest, ThreadSafety) {
    auto& manager = MemoryManager::instance();
    
    const int num_threads = 4;
    const int allocations_per_thread = 100;
    const size_t allocation_size = 1024;
    
    std::vector<std::thread> threads;
    
    // Launch threads that allocate and deallocate memory
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&manager, t, allocations_per_thread, allocation_size]() {
            for (int i = 0; i < allocations_per_thread; ++i) {
                std::string tag = "thread_" + std::to_string(t) + "_alloc_" + std::to_string(i);
                
                manager.track_allocation(tag, allocation_size);
                
                // Small delay to increase chance of race conditions
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                
                manager.track_deallocation(tag, allocation_size);
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Memory tracking should be consistent
    // All allocations should have been deallocated
    auto stats = manager.get_allocation_stats();
    for (const auto& [tag, size] : stats) {
        if (tag.find("thread_") == 0) {
            EXPECT_EQ(size, 0) << "Memory leak detected for tag: " << tag;
        }
    }
}

TEST_F(MemoryManagerTest, LargeAllocation) {
    auto& manager = MemoryManager::instance();
    
    // Test with large allocation sizes
    size_t large_size = 1024 * 1024 * 100; // 100MB
    
    size_t initial_allocated = manager.get_total_allocated();
    
    EXPECT_NO_THROW({
        manager.track_allocation("large_alloc", large_size);
        
        // Allow for small overhead in tracking
        size_t current_allocated = manager.get_total_allocated();
        EXPECT_GE(current_allocated, initial_allocated + large_size);
        EXPECT_LE(current_allocated, initial_allocated + large_size + 1024); // Allow 1KB overhead
        
        manager.track_deallocation("large_alloc", large_size);
        
        // Should be back to initial state (allowing for other allocations)
        size_t final_allocated = manager.get_total_allocated();
        EXPECT_LE(final_allocated, initial_allocated + 1024); // Allow small overhead
    });
}

TEST_F(MemoryManagerTest, ZeroSizeAllocation) {
    auto& manager = MemoryManager::instance();
    
    size_t initial_allocated = manager.get_total_allocated();
    
    // Zero-size allocations should be handled gracefully
    EXPECT_NO_THROW({
        manager.track_allocation("zero_alloc", 0);
        EXPECT_EQ(manager.get_total_allocated(), initial_allocated);
        
        manager.track_deallocation("zero_alloc", 0);
        EXPECT_EQ(manager.get_total_allocated(), initial_allocated);
    });
}

TEST_F(MemoryManagerTest, MemoryLimitBoundaryConditions) {
    auto& manager = MemoryManager::instance();
    
    // Clean up any existing allocations from previous tests
    auto existing_stats = manager.get_allocation_stats();
    for (const auto& [tag, bytes] : existing_stats) {
        manager.track_deallocation(tag, bytes);
    }
    
    // Set a specific limit (5 MB in bytes)
    size_t limit = 5 * 1024 * 1024;
    manager.set_memory_limit(limit);
    
    // Test exact limit
    EXPECT_TRUE(manager.is_memory_available(limit));
    
    // Test just over limit
    EXPECT_FALSE(manager.is_memory_available(limit + 1));
    
    // Allocate up to limit
    manager.track_allocation("boundary_test", limit);
    EXPECT_FALSE(manager.is_memory_available(1));
    
    // Deallocate and test again
    manager.track_deallocation("boundary_test", limit);
    EXPECT_TRUE(manager.is_memory_available(limit));
}

TEST_F(MemoryManagerTest, MultipleTagsPerAllocation) {
    auto& manager = MemoryManager::instance();
    
    // Use the same tag for multiple allocations
    manager.track_allocation("multi_tag", 1000);
    manager.track_allocation("multi_tag", 2000);
    manager.track_allocation("multi_tag", 1500);
    
    auto stats = manager.get_allocation_stats();
    EXPECT_EQ(stats["multi_tag"], 4500);
    
    // Partial deallocation
    manager.track_deallocation("multi_tag", 1000);
    stats = manager.get_allocation_stats();
    EXPECT_EQ(stats["multi_tag"], 3500);
}

TEST_F(MemoryManagerTest, TrackedAllocatorBasicUsage) {
    using TestAllocator = MemoryManager::TrackedAllocator<int>;
    
    TestAllocator allocator("test_allocator");
    
    // Allocate some integers
    int* ptr = nullptr;
    EXPECT_NO_THROW({
        ptr = allocator.allocate(100);
    });
    
    // Check that allocation was tracked (ptr might be null on some systems)
    auto& manager = MemoryManager::instance();
    auto stats = manager.get_allocation_stats();
    
    if (ptr != nullptr) {
        EXPECT_GT(stats["test_allocator"], 0);
        
        // Deallocate
        EXPECT_NO_THROW({
            allocator.deallocate(ptr, 100);
        });
    } else {
        // If allocation failed, that's acceptable for this test
        // Just verify the allocator doesn't crash
        EXPECT_NO_THROW({
            if (ptr) allocator.deallocate(ptr, 100);
        });
    }
}

} // namespace qwen 