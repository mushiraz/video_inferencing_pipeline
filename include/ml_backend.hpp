#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>

namespace qwen {

struct AnalysisConfig; // Forward declaration
struct AnalysisResult; // Forward declaration

class MLBackend {
public:
    explicit MLBackend(const AnalysisConfig& config);
    ~MLBackend();

    // Analyze frames using the ML model
    AnalysisResult analyze(const std::vector<cv::Mat>& frames, 
                          const std::string& prompt);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace qwen 