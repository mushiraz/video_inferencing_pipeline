#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

namespace qwen {

struct AnalysisConfig; // Forward declaration

class FrameExtractor {
public:
    explicit FrameExtractor(const AnalysisConfig& config);
    ~FrameExtractor();

    // Extract frames from video using configured strategy
    std::vector<cv::Mat> extract(const std::string& video_path);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace qwen 