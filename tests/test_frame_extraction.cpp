#include <gtest/gtest.h>
#include "frame_extractor.hpp"
#include "video_processor.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>

namespace qwen {

class FrameExtractionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test video file
        create_test_video();
        
        config_.max_frames = 8;
        config_.target_size = cv::Size(256, 256);
        config_.sampling_strategy = "uniform";
    }
    
    void TearDown() override {
        // Clean up test video
        std::remove("test_video.mp4");
    }
    
    void create_test_video() {
        cv::VideoWriter writer;
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        
        if (!writer.open("test_video.mp4", fourcc, 10.0, cv::Size(640, 480))) {
            FAIL() << "Could not create test video file";
        }
        
        // Create 50 frames with different colors
        for (int i = 0; i < 50; ++i) {
            cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
            
            // Create a gradient that changes over time
            int color_value = (i * 255) / 49;
            cv::rectangle(frame, cv::Point(0, 0), cv::Point(640, 480), 
                         cv::Scalar(color_value, 255 - color_value, 128), -1);
            
            // Add a moving circle
            int circle_x = (i * 600) / 49 + 20;
            cv::circle(frame, cv::Point(circle_x, 240), 30, cv::Scalar(255, 255, 255), -1);
            
            writer << frame;
        }
        writer.release();
    }
    
    AnalysisConfig config_;
};

TEST_F(FrameExtractionTest, BasicExtraction) {
    FrameExtractor extractor(config_);
    
    EXPECT_NO_THROW({
        auto frames = extractor.extract("test_video.mp4");
        EXPECT_FALSE(frames.empty());
        EXPECT_LE(frames.size(), static_cast<size_t>(config_.max_frames));
    });
}

TEST_F(FrameExtractionTest, UniformSampling) {
    config_.sampling_strategy = "uniform";
    FrameExtractor extractor(config_);
    
    auto frames = extractor.extract("test_video.mp4");
    
    EXPECT_EQ(frames.size(), static_cast<size_t>(config_.max_frames));
    
    // Check that frames are properly resized
    for (const auto& frame : frames) {
        EXPECT_EQ(frame.size(), config_.target_size);
        EXPECT_EQ(frame.channels(), 3); // RGB
    }
}

TEST_F(FrameExtractionTest, KeyframeSampling) {
    config_.sampling_strategy = "keyframe";
    FrameExtractor extractor(config_);
    
    auto frames = extractor.extract("test_video.mp4");
    
    EXPECT_FALSE(frames.empty());
    EXPECT_LE(frames.size(), static_cast<size_t>(config_.max_frames));
    
    // Check frame properties
    for (const auto& frame : frames) {
        EXPECT_EQ(frame.size(), config_.target_size);
        EXPECT_EQ(frame.channels(), 3);
    }
}

TEST_F(FrameExtractionTest, DifferentFrameCounts) {
    std::vector<int> frame_counts = {1, 4, 8, 16, 32};
    
    for (int count : frame_counts) {
        config_.max_frames = count;
        FrameExtractor extractor(config_);
        
        auto frames = extractor.extract("test_video.mp4");
        
        EXPECT_FALSE(frames.empty());
        EXPECT_LE(frames.size(), static_cast<size_t>(count));
        
        // For uniform sampling, we should get exactly the requested count
        if (config_.sampling_strategy == "uniform") {
            EXPECT_EQ(frames.size(), static_cast<size_t>(std::min(count, 50))); // 50 is total frames in test video
        }
    }
}

TEST_F(FrameExtractionTest, DifferentTargetSizes) {
    std::vector<cv::Size> sizes = {
        cv::Size(128, 128),
        cv::Size(256, 256),
        cv::Size(512, 512),
        cv::Size(224, 224)  // Common ML input size
    };
    
    for (const auto& size : sizes) {
        config_.target_size = size;
        FrameExtractor extractor(config_);
        
        auto frames = extractor.extract("test_video.mp4");
        
        EXPECT_FALSE(frames.empty());
        
        for (const auto& frame : frames) {
            EXPECT_EQ(frame.size(), size);
        }
    }
}

TEST_F(FrameExtractionTest, InvalidVideoPath) {
    FrameExtractor extractor(config_);
    
    EXPECT_THROW({
        extractor.extract("nonexistent_video.mp4");
    }, std::runtime_error);
}

TEST_F(FrameExtractionTest, VideoProcessorIntegration) {
    VideoProcessor processor(config_);
    
    // Test video info extraction
    auto info = processor.get_video_info("test_video.mp4");
    EXPECT_EQ(info.total_frames, 50);
    EXPECT_DOUBLE_EQ(info.fps, 10.0);
    EXPECT_EQ(info.frame_size, cv::Size(640, 480));
    
    // Test frame extraction through processor
    auto frames = processor.extract_frames("test_video.mp4");
    EXPECT_FALSE(frames.empty());
    EXPECT_LE(frames.size(), static_cast<size_t>(config_.max_frames));
}

TEST_F(FrameExtractionTest, FrameQuality) {
    FrameExtractor extractor(config_);
    auto frames = extractor.extract("test_video.mp4");
    
    EXPECT_FALSE(frames.empty());
    
    // Check that frames contain actual data (not all zeros)
    for (const auto& frame : frames) {
        cv::Scalar mean_color = cv::mean(frame);
        
        // At least one channel should have non-zero mean
        bool has_content = (mean_color[0] > 0) || (mean_color[1] > 0) || (mean_color[2] > 0);
        EXPECT_TRUE(has_content) << "Frame appears to be empty";
        
        // Check that pixel values are in valid range [0, 255]
        double min_val, max_val;
        cv::minMaxLoc(frame, &min_val, &max_val);
        EXPECT_GE(min_val, 0.0);
        EXPECT_LE(max_val, 255.0);
    }
}

TEST_F(FrameExtractionTest, ConsistentExtraction) {
    FrameExtractor extractor(config_);
    
    // Extract frames multiple times
    auto frames1 = extractor.extract("test_video.mp4");
    auto frames2 = extractor.extract("test_video.mp4");
    
    // Should get the same number of frames
    EXPECT_EQ(frames1.size(), frames2.size());
    
    // For uniform sampling, frames should be identical
    if (config_.sampling_strategy == "uniform") {
        EXPECT_EQ(frames1.size(), frames2.size());
        
        for (size_t i = 0; i < frames1.size(); ++i) {
            cv::Mat diff;
            cv::absdiff(frames1[i], frames2[i], diff);
            cv::Scalar mean_diff = cv::mean(diff);
            
            // Frames should be identical (very small difference allowed for floating point)
            EXPECT_LT(mean_diff[0] + mean_diff[1] + mean_diff[2], 1.0);
        }
    }
}

TEST_F(FrameExtractionTest, MemoryEfficiency) {
    // Test with larger frame count to check memory usage
    config_.max_frames = 32;
    config_.target_size = cv::Size(512, 512);
    
    FrameExtractor extractor(config_);
    
    // This should not cause memory issues
    EXPECT_NO_THROW({
        auto frames = extractor.extract("test_video.mp4");
        
        // Calculate approximate memory usage
        size_t memory_per_frame = frames[0].total() * frames[0].elemSize();
        size_t total_memory = frames.size() * memory_per_frame;
        
        // Should be reasonable (less than 100MB for this test)
        EXPECT_LT(total_memory, 100 * 1024 * 1024);
    });
}

} // namespace qwen 