#include "compute_manager.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace qwen {

// CPU Backend Implementation
CPUComputeBackend::CPUComputeBackend() {
    std::cout << "Initialized CPU compute backend" << std::endl;
}

void CPUComputeBackend::process_frames_gpu(const std::vector<cv::Mat>& frames,
                                         std::vector<float>& output) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Estimate output size
    size_t total_elements = 0;
    for (const auto& frame : frames) {
        total_elements += frame.total() * frame.channels();
    }
    
    output.clear();
    output.reserve(total_elements);
    
    // Simple CPU processing - convert frames to float array
    for (const auto& frame : frames) {
        cv::Mat float_frame;
        frame.convertTo(float_frame, CV_32F, 1.0/255.0);
        
        // Flatten the frame
        cv::Mat flat = float_frame.reshape(1, float_frame.total() * float_frame.channels());
        std::vector<float> frame_data = flat.isContinuous() ? flat : flat.clone();
        
        output.insert(output.end(), frame_data.begin(), frame_data.end());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_execution_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
}

std::string CPUComputeBackend::get_device_name() const {
    return "CPU";
}

size_t CPUComputeBackend::get_memory_available() const {
#ifdef __APPLE__
    int mib[2];
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    
    int64_t physical_memory;
    size_t length = sizeof(int64_t);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    
    return static_cast<size_t>(physical_memory);
#else
    return 8ULL * 1024 * 1024 * 1024; // Default 8GB
#endif
}

bool CPUComputeBackend::is_available() const {
    return true; // CPU is always available
}

double CPUComputeBackend::get_last_execution_time() const {
    return last_execution_time_;
}

size_t CPUComputeBackend::get_memory_usage() const {
    // Simple estimation based on recent processing
    return memory_usage_;
}

// ComputeManager Implementation
ComputeManager& ComputeManager::instance() {
    static ComputeManager instance;
    return instance;
}

ComputeManager::ComputeManager() {
    initialize_backends();
}

void ComputeManager::initialize_backends() {
    // Always add CPU backend
    available_backends_.push_back("CPU");
    
#ifdef USE_METAL
    // Check for Metal availability
    if (is_metal_available()) {
        available_backends_.push_back("Metal");
    }
#endif

#ifdef USE_HIP
    // Check for HIP availability
    if (is_hip_available()) {
        available_backends_.push_back("HIP");
    }
#endif

    std::cout << "ComputeManager initialized with backends: ";
    for (size_t i = 0; i < available_backends_.size(); ++i) {
        std::cout << available_backends_[i];
        if (i < available_backends_.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

std::unique_ptr<ComputeBackend> ComputeManager::create_backend(const std::string& backend_type) {
    if (backend_type == "CPU") {
        return std::make_unique<CPUComputeBackend>();
    }
    
#ifdef USE_METAL
    if (backend_type == "Metal" && is_metal_available()) {
        return create_metal_backend();
    }
#endif

#ifdef USE_HIP
    if (backend_type == "HIP" && is_hip_available()) {
        return create_hip_backend();
    }
#endif

    // Fallback to CPU
    std::cout << "Backend " << backend_type << " not available, falling back to CPU" << std::endl;
    return std::make_unique<CPUComputeBackend>();
}

std::unique_ptr<ComputeBackend> ComputeManager::create_best_backend() {
    // Priority order: Metal (on macOS), HIP (on AMD), CPU
    
#ifdef USE_METAL
    if (is_metal_available()) {
        return create_backend("Metal");
    }
#endif

#ifdef USE_HIP
    if (is_hip_available()) {
        return create_backend("HIP");
    }
#endif

    return create_backend("CPU");
}

std::vector<std::string> ComputeManager::get_available_backends() const {
    return available_backends_;
}

bool ComputeManager::is_backend_available(const std::string& backend_type) const {
    return std::find(available_backends_.begin(), available_backends_.end(), backend_type) 
           != available_backends_.end();
}

#ifdef USE_HIP
bool ComputeManager::is_hip_available() const {
    // Check if HIP runtime is available
    // This is a simplified check - in practice you'd use hipGetDeviceCount()
    return false; // Placeholder - implement actual HIP detection
}

std::unique_ptr<ComputeBackend> ComputeManager::create_hip_backend() {
    // Placeholder for HIP backend creation
    return std::make_unique<CPUComputeBackend>(); // Fallback for now
}
#endif

} // namespace qwen

// Metal implementation - must be outside namespace for Objective-C++
#ifdef USE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

namespace qwen {

class MetalComputeBackend : public ComputeBackend {
private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    double last_execution_time_;
    size_t memory_usage_;

public:
    MetalComputeBackend() : last_execution_time_(0.0), memory_usage_(0) {
        device_ = MTLCreateSystemDefaultDevice();
        if (device_) {
            command_queue_ = [device_ newCommandQueue];
            std::cout << "Initialized Metal compute backend" << std::endl;
        } else {
            throw std::runtime_error("Failed to create Metal device");
        }
    }
    
    ~MetalComputeBackend() {
        // ARC will handle cleanup
    }
    
    void process_frames_gpu(const std::vector<cv::Mat>& frames,
                           std::vector<float>& output) override {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // For now, implement a simple Metal compute shader
        // In practice, this would use actual Metal compute shaders
        
        // Estimate output size
        size_t total_elements = 0;
        for (const auto& frame : frames) {
            total_elements += frame.total() * frame.channels();
        }
        
        output.clear();
        output.reserve(total_elements);
        
        // Create Metal buffers and process
        @autoreleasepool {
            // Convert frames to Metal buffers
            for (const auto& frame : frames) {
                cv::Mat float_frame;
                frame.convertTo(float_frame, CV_32F, 1.0/255.0);
                
                // Create Metal buffer
                size_t buffer_size = float_frame.total() * float_frame.channels() * sizeof(float);
                id<MTLBuffer> input_buffer = [device_ newBufferWithBytes:float_frame.data
                                                                  length:buffer_size
                                                                 options:MTLResourceStorageModeShared];
                
                // For this example, just copy the data back
                // In practice, you'd run compute shaders here
                float* buffer_data = (float*)[input_buffer contents];
                size_t num_elements = buffer_size / sizeof(float);
                
                for (size_t i = 0; i < num_elements; ++i) {
                    output.push_back(buffer_data[i]);
                }
                
                memory_usage_ += buffer_size;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        last_execution_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
    }
    
    std::string get_device_name() const override {
        if (device_) {
            NSString* name = [device_ name];
            return std::string([name UTF8String]);
        }
        return "Metal (Unknown Device)";
    }
    
    size_t get_memory_available() const override {
        if (device_) {
            return [device_ recommendedMaxWorkingSetSize];
        }
        return 0;
    }
    
    bool is_available() const override {
        return device_ != nil;
    }
    
    double get_last_execution_time() const override {
        return last_execution_time_;
    }
    
    size_t get_memory_usage() const override {
        return memory_usage_;
    }
};

bool ComputeManager::is_metal_available() const {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
}

std::unique_ptr<ComputeBackend> ComputeManager::create_metal_backend() {
    return std::make_unique<MetalComputeBackend>();
}

} // namespace qwen

#else

namespace qwen {

bool ComputeManager::is_metal_available() const {
    return false;
}

std::unique_ptr<ComputeBackend> ComputeManager::create_metal_backend() {
    return std::make_unique<CPUComputeBackend>();
}

} // namespace qwen

#endif 