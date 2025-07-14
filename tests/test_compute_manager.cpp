#include <gtest/gtest.h>
#include "compute_manager.hpp"
#include <opencv2/opencv.hpp>

namespace qwen {

class ComputeManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test frames
        test_frames_.clear();
        for (int i = 0; i < 4; ++i) {
            cv::Mat frame = cv::Mat::zeros(256, 256, CV_8UC3);
            cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            test_frames_.push_back(frame);
        }
    }
    
    std::vector<cv::Mat> test_frames_;
};

TEST_F(ComputeManagerTest, SingletonInstance) {
    auto& manager1 = ComputeManager::instance();
    auto& manager2 = ComputeManager::instance();
    
    EXPECT_EQ(&manager1, &manager2);
}

TEST_F(ComputeManagerTest, AvailableBackends) {
    auto& manager = ComputeManager::instance();
    auto backends = manager.get_available_backends();
    
    EXPECT_FALSE(backends.empty());
    
    // CPU should always be available
    bool cpu_found = false;
    for (const auto& backend : backends) {
        if (backend == "CPU") {
            cpu_found = true;
            break;
        }
    }
    EXPECT_TRUE(cpu_found);
}

TEST_F(ComputeManagerTest, CreateBestBackend) {
    auto& manager = ComputeManager::instance();
    
    auto backend = manager.create_best_backend();
    EXPECT_NE(backend, nullptr);
    EXPECT_TRUE(backend->is_available());
    EXPECT_FALSE(backend->get_device_name().empty());
}

TEST_F(ComputeManagerTest, CreateSpecificBackend) {
    auto& manager = ComputeManager::instance();
    
    // CPU backend should always work
    auto cpu_backend = manager.create_backend("CPU");
    EXPECT_NE(cpu_backend, nullptr);
    EXPECT_TRUE(cpu_backend->is_available());
    EXPECT_EQ(cpu_backend->get_device_name(), "CPU");
}

TEST_F(ComputeManagerTest, CPUBackendBasicFunctionality) {
    auto& manager = ComputeManager::instance();
    auto backend = manager.create_backend("CPU");
    
    EXPECT_NE(backend, nullptr);
    EXPECT_TRUE(backend->is_available());
    EXPECT_GT(backend->get_memory_available(), 0);
    EXPECT_FALSE(backend->get_device_name().empty());
}

TEST_F(ComputeManagerTest, FrameProcessing) {
    auto& manager = ComputeManager::instance();
    auto backend = manager.create_backend("CPU");
    
    std::vector<float> output;
    EXPECT_NO_THROW({
        backend->process_frames_gpu(test_frames_, output);
    });
    
    EXPECT_FALSE(output.empty());
    EXPECT_GT(backend->get_last_execution_time(), 0.0);
}

TEST_F(ComputeManagerTest, BackendAvailabilityCheck) {
    auto& manager = ComputeManager::instance();
    
    EXPECT_TRUE(manager.is_backend_available("CPU"));
    EXPECT_FALSE(manager.is_backend_available("NonExistentBackend"));
}

TEST_F(ComputeManagerTest, EmptyFrameProcessing) {
    auto& manager = ComputeManager::instance();
    auto backend = manager.create_backend("CPU");
    
    std::vector<cv::Mat> empty_frames;
    std::vector<float> output;
    
    EXPECT_NO_THROW({
        backend->process_frames_gpu(empty_frames, output);
    });
    
    EXPECT_TRUE(output.empty());
}

TEST_F(ComputeManagerTest, LargeFrameProcessing) {
    auto& manager = ComputeManager::instance();
    auto backend = manager.create_backend("CPU");
    
    // Create larger frames
    std::vector<cv::Mat> large_frames;
    for (int i = 0; i < 2; ++i) {
        cv::Mat frame = cv::Mat::zeros(1024, 1024, CV_8UC3);
        cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        large_frames.push_back(frame);
    }
    
    std::vector<float> output;
    EXPECT_NO_THROW({
        backend->process_frames_gpu(large_frames, output);
    });
    
    EXPECT_FALSE(output.empty());
    EXPECT_GT(backend->get_last_execution_time(), 0.0);
}

TEST_F(ComputeManagerTest, MultipleBackendCreation) {
    auto& manager = ComputeManager::instance();
    
    auto backend1 = manager.create_backend("CPU");
    auto backend2 = manager.create_backend("CPU");
    
    EXPECT_NE(backend1, nullptr);
    EXPECT_NE(backend2, nullptr);
    EXPECT_NE(backend1.get(), backend2.get()); // Different instances
}

TEST_F(ComputeManagerTest, ConsistentProcessing) {
    auto& manager = ComputeManager::instance();
    auto backend = manager.create_backend("CPU");
    
    std::vector<float> output1, output2;
    
    backend->process_frames_gpu(test_frames_, output1);
    backend->process_frames_gpu(test_frames_, output2);
    
    // Results should be consistent for the same input
    EXPECT_EQ(output1.size(), output2.size());
    
    // Allow for small floating point differences
    for (size_t i = 0; i < output1.size(); ++i) {
        EXPECT_NEAR(output1[i], output2[i], 1e-6);
    }
}

#ifdef USE_METAL
TEST_F(ComputeManagerTest, MetalBackendAvailability) {
    auto& manager = ComputeManager::instance();
    
    if (manager.is_backend_available("Metal")) {
        auto backend = manager.create_backend("Metal");
        EXPECT_NE(backend, nullptr);
        EXPECT_TRUE(backend->is_available());
        
        // Check that device name is not empty and contains expected content
        std::string device_name = backend->get_device_name();
        EXPECT_FALSE(device_name.empty());
        
        // The device name should contain either "Metal" or be a GPU name
        bool valid_name = (device_name.find("Metal") != std::string::npos) ||
                         (device_name.find("AMD") != std::string::npos) ||
                         (device_name.find("Intel") != std::string::npos) ||
                         (device_name.find("Apple") != std::string::npos) ||
                         (device_name.find("GPU") != std::string::npos);
        EXPECT_TRUE(valid_name) << "Device name: " << device_name;
    }
}
#endif

} // namespace qwen 