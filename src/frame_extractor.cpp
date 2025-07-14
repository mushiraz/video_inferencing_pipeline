#include "frame_extractor.hpp"
#include "video_processor.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <execution>
#include <chrono>
#include <iostream>

namespace qwen {

class FrameExtractor::Impl {
public:
    explicit Impl(const AnalysisConfig& config) : config_(config) {}

    std::vector<cv::Mat> extract_uniform(cv::VideoCapture& cap, int total_frames) {
        std::vector<int> frame_indices(static_cast<size_t>(config_.max_frames));
        std::iota(frame_indices.begin(), frame_indices.end(), 0);
        
        std::transform(frame_indices.begin(), frame_indices.end(), frame_indices.begin(),
            [total_frames, this](int idx) {
                return static_cast<int>((static_cast<double>(idx) / (config_.max_frames - 1)) * (total_frames - 1));
            });

        return extract_frames_by_indices(cap, frame_indices);
    }

    std::vector<cv::Mat> extract_keyframes(cv::VideoCapture& cap, int total_frames) {
        std::vector<int> keyframe_indices;
        cv::Mat prev_frame, curr_frame;
        double threshold = 30.0;

        // Always include first frame
        keyframe_indices.push_back(0);
        
        int sample_rate = std::max(1, total_frames / (config_.max_frames * 3));
        
        for (int i = sample_rate; i < total_frames && static_cast<int>(keyframe_indices.size()) < config_.max_frames; i += sample_rate) {
            cap.set(cv::CAP_PROP_POS_FRAMES, i);
            if (!cap.read(curr_frame)) continue;
            
            if (!prev_frame.empty()) {
                cv::Mat diff;
                cv::absdiff(prev_frame, curr_frame, diff);
                cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
                
                double mean_diff = cv::mean(diff)[0];
                if (mean_diff > threshold) {
                    keyframe_indices.push_back(i);
                }
            }
            
            prev_frame = curr_frame.clone();
        }

        // Fill remaining slots uniformly if needed
        while (static_cast<int>(keyframe_indices.size()) < config_.max_frames) {
            int next_idx = keyframe_indices.back() + total_frames / config_.max_frames;
            if (next_idx >= total_frames) break;
            keyframe_indices.push_back(next_idx);
        }

        return extract_frames_by_indices(cap, keyframe_indices);
    }

    std::vector<cv::Mat> extract_frames_by_indices(cv::VideoCapture& cap, 
                                                  const std::vector<int>& indices) {
        std::vector<cv::Mat> frames;
        frames.reserve(indices.size());

        for (int idx : indices) {
            cap.set(cv::CAP_PROP_POS_FRAMES, idx);
            cv::Mat frame;
            if (cap.read(frame)) {
                // Resize for memory efficiency
                if (frame.size() != config_.target_size) {
                    cv::resize(frame, frame, config_.target_size, 0, 0, cv::INTER_LINEAR);
                }
                
                // Convert BGR to RGB for model input
                cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
                frames.push_back(frame.clone());
            }
        }

        return frames;
    }

    const AnalysisConfig& get_config() const { return config_; }

private:
    AnalysisConfig config_;
};

FrameExtractor::FrameExtractor(const AnalysisConfig& config) 
    : pimpl_(std::make_unique<Impl>(config)) {}

FrameExtractor::~FrameExtractor() = default;

std::vector<cv::Mat> FrameExtractor::extract(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + video_path);
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Extracting frames from " << video_path 
              << " (total frames: " << total_frames 
              << ", strategy: " << pimpl_->get_config().sampling_strategy << ")" << std::endl;
    
    if (pimpl_->get_config().sampling_strategy == "keyframe") {
        return pimpl_->extract_keyframes(cap, total_frames);
    } else {
        return pimpl_->extract_uniform(cap, total_frames);
    }
}

} // namespace qwen 