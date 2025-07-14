#pragma once

#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace qwen {

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    
    // Core GPU processing interface
    virtual void process_frames_gpu(const std::vector<cv::Mat>& frames,
                                   std::vector<float>& output) = 0;
    
    // Device information
    virtual std::string get_device_name() const = 0;
    virtual size_t get_memory_available() const = 0;
    virtual bool is_available() const = 0;
    
    // Performance monitoring
    virtual double get_last_execution_time() const = 0;
    virtual size_t get_memory_usage() const = 0;
};

class CPUComputeBackend : public ComputeBackend {
public:
    CPUComputeBackend();
    ~CPUComputeBackend() override = default;
    
    void process_frames_gpu(const std::vector<cv::Mat>& frames,
                           std::vector<float>& output) override;
    
    std::string get_device_name() const override;
    size_t get_memory_available() const override;
    bool is_available() const override;
    double get_last_execution_time() const override;
    size_t get_memory_usage() const override;

private:
    mutable double last_execution_time_ = 0.0;
    mutable size_t memory_usage_ = 0;
};

class ComputeManager {
public:
    static ComputeManager& instance();
    
    // Backend management
    std::unique_ptr<ComputeBackend> create_backend(const std::string& backend_type);
    std::unique_ptr<ComputeBackend> create_best_backend();
    std::vector<std::string> get_available_backends() const;
    bool is_backend_available(const std::string& backend_type) const;

private:
    ComputeManager();
    
    void initialize_backends();
    
    // Platform-specific backend creation
    bool is_metal_available() const;
    std::unique_ptr<ComputeBackend> create_metal_backend();
    
#ifdef USE_HIP
    bool is_hip_available() const;
    std::unique_ptr<ComputeBackend> create_hip_backend();
#endif
    
    std::vector<std::string> available_backends_;
};

} // namespace qwen 