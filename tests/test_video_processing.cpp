#include <gtest/gtest.h>
#include "video_processor.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace qwen {

class VideoProcessingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test video files
        create_test_video("test_video_short.mp4", 20, cv::Size(320, 240));
        create_test_video("test_video_long.mp4", 100, cv::Size(640, 480));
        
        // Default configuration
        config_.max_frames = 8;
        config_.target_size = cv::Size(256, 256);
        config_.sampling_strategy = "uniform";
        config_.use_gpu = false; // Use CPU for consistent testing
        config_.num_threads = 2;
        config_.max_memory_mb = 1024;
    }
    
    void TearDown() override {
        // Clean up test videos
        std::remove("test_video_short.mp4");
        std::remove("test_video_long.mp4");
    }
    
    void create_test_video(const std::string& filename, int num_frames, cv::Size size) {
        cv::VideoWriter writer;
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        
        if (!writer.open(filename, fourcc, 15.0, size)) {
            FAIL() << "Could not create test video: " << filename;
        }
        
        for (int i = 0; i < num_frames; ++i) {
            cv::Mat frame = cv::Mat::zeros(size.height, size.width, CV_8UC3);
            
            // Create distinctive patterns for each frame
            int color_shift = (i * 255) / (num_frames - 1);
            cv::rectangle(frame, cv::Point(0, 0), cv::Point(size.width, size.height), 
                         cv::Scalar(color_shift, 255 - color_shift, 128), -1);
            
            // Add frame number as visual indicator
            cv::putText(frame, std::to_string(i), cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            
            writer << frame;
        }
        writer.release();
    }
    
    AnalysisConfig config_;
};

TEST_F(VideoProcessingTest, BasicVideoInfo) {
    VideoProcessor processor(config_);
    
    auto info = processor.get_video_info("test_video_short.mp4");
    
    EXPECT_EQ(info.total_frames, 20);
    EXPECT_DOUBLE_EQ(info.fps, 15.0);
    EXPECT_EQ(info.frame_size, cv::Size(320, 240));
    EXPECT_FALSE(info.codec.empty());
}

TEST_F(VideoProcessingTest, FrameExtraction) {
    VideoProcessor processor(config_);
    
    auto frames = processor.extract_frames("test_video_short.mp4");
    
    EXPECT_FALSE(frames.empty());
    EXPECT_LE(frames.size(), static_cast<size_t>(config_.max_frames));
    
    // Check frame properties
    for (const auto& frame : frames) {
        EXPECT_EQ(frame.size(), config_.target_size);
        EXPECT_EQ(frame.channels(), 3);
    }
}

TEST_F(VideoProcessingTest, BasicVideoAnalysis) {
    VideoProcessor processor(config_);
    
    auto result = processor.analyze_video("test_video_short.mp4");
    
    EXPECT_FALSE(result.description.empty());
    EXPECT_FALSE(result.objects.empty());
    EXPECT_FALSE(result.actions.empty());
    EXPECT_FALSE(result.scene_context.empty());
    EXPECT_GE(result.confidence_score, 0.0);
    EXPECT_LE(result.confidence_score, 1.0);
    EXPECT_GT(result.processing_time.count(), 0);
}

TEST_F(VideoProcessingTest, CustomPromptAnalysis) {
    VideoProcessor processor(config_);
    
    std::string custom_prompt = "Focus on detecting movement and color changes";
    auto result = processor.analyze_video("test_video_short.mp4", custom_prompt);
    
    EXPECT_FALSE(result.description.empty());
    EXPECT_GT(result.processing_time.count(), 0);
    
    // The description should somehow reflect the custom prompt
    // (This depends on the ML backend implementation)
}

TEST_F(VideoProcessingTest, DifferentFrameCounts) {
    std::vector<int> frame_counts = {4, 8, 16};
    
    for (int count : frame_counts) {
        config_.max_frames = count;
        VideoProcessor processor(config_);
        
        auto result = processor.analyze_video("test_video_long.mp4");
        
        EXPECT_FALSE(result.description.empty());
        EXPECT_GT(result.processing_time.count(), 0);
        
        // More frames might take longer (though not guaranteed with placeholder)
        // This is more of a sanity check
    }
}

TEST_F(VideoProcessingTest, DifferentSamplingStrategies) {
    std::vector<std::string> strategies = {"uniform", "keyframe"};
    
    for (const auto& strategy : strategies) {
        config_.sampling_strategy = strategy;
        VideoProcessor processor(config_);
        
        auto result = processor.analyze_video("test_video_long.mp4");
        
        EXPECT_FALSE(result.description.empty());
        EXPECT_GT(result.processing_time.count(), 0);
    }
}

TEST_F(VideoProcessingTest, BatchProcessing) {
    VideoProcessor processor(config_);
    
    std::vector<std::string> video_paths = {
        "test_video_short.mp4",
        "test_video_long.mp4"
    };
    
    auto results = processor.batch_analyze(video_paths);
    
    EXPECT_EQ(results.size(), 2);
    
    for (const auto& [path, result] : results) {
        EXPECT_TRUE(path == "test_video_short.mp4" || path == "test_video_long.mp4");
        EXPECT_FALSE(result.description.empty());
        EXPECT_GT(result.processing_time.count(), 0);
    }
}

TEST_F(VideoProcessingTest, StreamingAnalysis) {
    VideoProcessor processor(config_);
    
    std::vector<AnalysisResult> streaming_results;
    
    processor.start_stream_analysis("test_video_long.mp4", 
        [&streaming_results](const AnalysisResult& result) {
            streaming_results.push_back(result);
        });
    
    EXPECT_FALSE(streaming_results.empty());
    
    for (const auto& result : streaming_results) {
        EXPECT_FALSE(result.description.empty());
    }
}

TEST_F(VideoProcessingTest, InvalidVideoHandling) {
    VideoProcessor processor(config_);
    
    EXPECT_THROW({
        processor.get_video_info("nonexistent_video.mp4");
    }, std::runtime_error);
    
    EXPECT_THROW({
        processor.extract_frames("nonexistent_video.mp4");
    }, std::runtime_error);
    
    // analyze_video should handle errors gracefully
    auto result = processor.analyze_video("nonexistent_video.mp4");
    EXPECT_EQ(result.confidence_score, 0.0);
    EXPECT_TRUE(result.description.find("failed") != std::string::npos);
}

TEST_F(VideoProcessingTest, MemoryLimitRespected) {
    // Set a very low memory limit
    config_.max_memory_mb = 100;
    VideoProcessor processor(config_);
    
    // Should still work but might trigger garbage collection
    EXPECT_NO_THROW({
        auto result = processor.analyze_video("test_video_short.mp4");
        EXPECT_FALSE(result.description.empty());
    });
}

TEST_F(VideoProcessingTest, ThreadingConfiguration) {
    std::vector<int> thread_counts = {1, 2, 4};
    
    for (int threads : thread_counts) {
        config_.num_threads = threads;
        VideoProcessor processor(config_);
        
        auto result = processor.analyze_video("test_video_short.mp4");
        
        EXPECT_FALSE(result.description.empty());
        EXPECT_GT(result.processing_time.count(), 0);
    }
}

TEST_F(VideoProcessingTest, GPUvsCSPUComparison) {
    // Test both GPU and CPU modes
    std::vector<bool> gpu_settings = {false, true};
    
    for (bool use_gpu : gpu_settings) {
        config_.use_gpu = use_gpu;
        VideoProcessor processor(config_);
        
        auto result = processor.analyze_video("test_video_short.mp4");
        
        EXPECT_FALSE(result.description.empty());
        EXPECT_GT(result.processing_time.count(), 0);
        
        // Both should produce valid results
        EXPECT_GE(result.confidence_score, 0.0);
        EXPECT_LE(result.confidence_score, 1.0);
    }
}

TEST_F(VideoProcessingTest, ResultConsistency) {
    VideoProcessor processor(config_);
    
    // Analyze the same video multiple times
    auto result1 = processor.analyze_video("test_video_short.mp4");
    auto result2 = processor.analyze_video("test_video_short.mp4");
    
    // Results should be consistent (for deterministic analysis)
    EXPECT_EQ(result1.objects.size(), result2.objects.size());
    EXPECT_EQ(result1.actions.size(), result2.actions.size());
    
    // Confidence scores should be similar (allowing for small variations)
    EXPECT_NEAR(result1.confidence_score, result2.confidence_score, 0.1);
}

TEST_F(VideoProcessingTest, LargeVideoHandling) {
    // Test with the longer video
    VideoProcessor processor(config_);
    
    auto result = processor.analyze_video("test_video_long.mp4");
    
    EXPECT_FALSE(result.description.empty());
    EXPECT_GT(result.processing_time.count(), 0);
    
    // Should handle longer videos without issues
    EXPECT_GE(result.confidence_score, 0.0);
}

TEST_F(VideoProcessingTest, ConfigurationValidation) {
    // Test with various configuration edge cases
    
    // Zero frames (should be handled gracefully)
    config_.max_frames = 0;
    EXPECT_NO_THROW({
        VideoProcessor processor(config_);
    });
    
    // Very large frame count
    config_.max_frames = 1000;
    EXPECT_NO_THROW({
        VideoProcessor processor(config_);
        auto result = processor.analyze_video("test_video_short.mp4");
        // Should still work, just limited by actual video length
    });
    
    // Very small target size
    config_.target_size = cv::Size(32, 32);
    EXPECT_NO_THROW({
        VideoProcessor processor(config_);
        auto result = processor.analyze_video("test_video_short.mp4");
    });
}

TEST_F(VideoProcessingTest, PerformanceMetrics) {
    VideoProcessor processor(config_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = processor.analyze_video("test_video_short.mp4");
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    // Processing time should be reasonable
    EXPECT_GT(result.processing_time.count(), 0);
    EXPECT_LT(result.processing_time.count(), 10000); // Less than 10 seconds
    
    // Total time should be close to processing time
    EXPECT_LE(result.processing_time.count(), total_time.count() + 100); // Allow 100ms overhead
}

} // namespace qwen 