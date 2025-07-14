#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>
#include <future>
#include <map>
#include <functional>
#include <chrono>
#include <nlohmann/json.hpp>

namespace qwen {

struct VideoInfo {
    int total_frames;
    double fps;
    double duration;
    cv::Size frame_size;
    std::string codec;
};

struct AnalysisConfig {
    int max_frames = 16;
    cv::Size target_size{512, 512};
    std::string sampling_strategy = "uniform"; // uniform, keyframe, adaptive
    bool use_gpu = true;
    int num_threads = std::thread::hardware_concurrency();
    size_t max_memory_mb = 8192;
};

struct AnalysisResult {
    std::string description;
    std::vector<std::string> objects;
    std::vector<std::string> actions;
    std::string scene_context;
    std::vector<std::string> temporal_events;
    double confidence_score;
    std::chrono::milliseconds processing_time;
};

class VideoProcessor {
public:
    explicit VideoProcessor(const AnalysisConfig& config = {});
    ~VideoProcessor();

    // Core functionality
    VideoInfo get_video_info(const std::string& video_path);
    std::vector<cv::Mat> extract_frames(const std::string& video_path);
    AnalysisResult analyze_video(const std::string& video_path, 
                               const std::string& custom_prompt = "");
    
    // Batch processing
    std::map<std::string, AnalysisResult> batch_analyze(
        const std::vector<std::string>& video_paths);
    
    // Streaming analysis
    void start_stream_analysis(const std::string& video_path,
                              std::function<void(const AnalysisResult&)> callback);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace qwen 