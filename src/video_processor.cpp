#include "video_processor.hpp"
#include "frame_extractor.hpp"
#include "ml_backend.hpp"
#include "memory_manager.hpp"
#include "compute_manager.hpp"
#include <thread>
#include <future>
#include <algorithm>
#include <execution>
#include <iostream>

namespace qwen {

class VideoProcessor::Impl {
public:
    explicit Impl(const AnalysisConfig& config) 
        : config_(config)
        , frame_extractor_(config)
        , ml_backend_(config)
        , memory_manager_(MemoryManager::instance())
        , compute_manager_(ComputeManager::instance()) {
        
        memory_manager_.set_memory_limit(config_.max_memory_mb * 1024 * 1024);
        
        // Initialize compute backend
        try {
            if (config_.use_gpu) {
                compute_backend_ = compute_manager_.create_best_backend();
                std::cout << "Using compute backend: " << compute_backend_->get_device_name() << std::endl;
            } else {
                compute_backend_ = compute_manager_.create_backend("CPU");
                std::cout << "Using CPU compute backend" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Failed to initialize compute backend: " << e.what() << std::endl;
            std::cout << "Falling back to CPU backend" << std::endl;
            compute_backend_ = compute_manager_.create_backend("CPU");
        }
        
        std::cout << "VideoProcessor initialized with:" << std::endl;
        std::cout << "  Max frames: " << config_.max_frames << std::endl;
        std::cout << "  Target size: " << config_.target_size.width << "x" << config_.target_size.height << std::endl;
        std::cout << "  Sampling strategy: " << config_.sampling_strategy << std::endl;
        std::cout << "  Threads: " << config_.num_threads << std::endl;
        std::cout << "  Memory limit: " << config_.max_memory_mb << "MB" << std::endl;
    }
    
    VideoInfo get_video_info(const std::string& video_path) {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open video file: " + video_path);
        }
        
        VideoInfo info;
        info.total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        info.fps = cap.get(cv::CAP_PROP_FPS);
        info.duration = info.total_frames / info.fps;
        info.frame_size = cv::Size(
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
        
        // Get codec information
        int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
        char codec_chars[5];
        codec_chars[0] = fourcc & 0xFF;
        codec_chars[1] = (fourcc >> 8) & 0xFF;
        codec_chars[2] = (fourcc >> 16) & 0xFF;
        codec_chars[3] = (fourcc >> 24) & 0xFF;
        codec_chars[4] = '\0';
        info.codec = std::string(codec_chars);
        
        return info;
    }
    
    std::vector<cv::Mat> extract_frames(const std::string& video_path) {
        return frame_extractor_.extract(video_path);
    }
    
    AnalysisResult analyze_video(const std::string& video_path, 
                                const std::string& custom_prompt = "") {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Extract frames
            auto frames = frame_extractor_.extract(video_path);
            if (frames.empty()) {
                throw std::runtime_error("No frames extracted from video");
            }
            
            std::cout << "Extracted " << frames.size() << " frames for analysis" << std::endl;
            
            // Preprocess frames using compute backend if available
            std::vector<float> processed_data;
            if (compute_backend_) {
                auto compute_start = std::chrono::high_resolution_clock::now();
                
                try {
                    compute_backend_->process_frames_gpu(frames, processed_data);
                    
                    auto compute_end = std::chrono::high_resolution_clock::now();
                    double compute_time = std::chrono::duration<double, std::milli>(
                        compute_end - compute_start).count();
                    
                    std::cout << "GPU preprocessing completed in " << compute_time << "ms" << std::endl;
                    
                } catch (const std::exception& e) {
                    std::cout << "GPU preprocessing failed: " << e.what() 
                              << ", continuing with original frames" << std::endl;
                }
            }
            
            // Analyze using ML backend
            std::string prompt = custom_prompt.empty() ? get_default_prompt() : custom_prompt;
            auto result = ml_backend_.analyze(frames, prompt);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
            
            return result;
            
        } catch (const std::exception& e) {
            AnalysisResult error_result;
            error_result.description = "Analysis failed: " + std::string(e.what());
            error_result.confidence_score = 0.0;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            error_result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
            
            return error_result;
        }
    }
    
    std::map<std::string, AnalysisResult> batch_analyze(
        const std::vector<std::string>& video_paths) {
        
        std::map<std::string, AnalysisResult> results;
        
        // Determine optimal batch size based on available memory
        size_t batch_size = calculate_optimal_batch_size(video_paths.size());
        
        std::cout << "Processing " << video_paths.size() << " videos in batches of " 
                  << batch_size << std::endl;
        
        for (size_t i = 0; i < video_paths.size(); i += batch_size) {
            size_t end_idx = std::min(i + batch_size, video_paths.size());
            
            // Process batch in parallel
            std::vector<std::future<std::pair<std::string, AnalysisResult>>> futures;
            
            for (size_t j = i; j < end_idx; ++j) {
                futures.emplace_back(
                    std::async(std::launch::async, [this, &video_paths, j]() {
                        return std::make_pair(
                            video_paths[j], 
                            analyze_video(video_paths[j], "")
                        );
                    })
                );
            }
            
            // Collect results
            for (auto& future : futures) {
                auto [path, result] = future.get();
                results[path] = std::move(result);
                
                std::cout << "Completed: " << path << " (confidence: " 
                          << result.confidence_score << ")" << std::endl;
            }
            
            // Memory cleanup between batches
            memory_manager_.garbage_collect();
        }
        
        return results;
    }
    
    void start_stream_analysis(const std::string& video_path,
                              std::function<void(const AnalysisResult&)> callback) {
        // Implementation for streaming analysis
        // This would process video in chunks and call callback for each result
        
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open video for streaming: " + video_path);
        }
        
        size_t chunk_size = static_cast<size_t>(config_.max_frames);
        std::vector<cv::Mat> chunk_buffer;
        chunk_buffer.reserve(chunk_size);
        
        cv::Mat frame;
        int frame_count = 0;
        
        std::cout << "Starting streaming analysis of " << video_path << std::endl;
        
        while (cap.read(frame)) {
            // Resize frame
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, config_.target_size);
            cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);
            
            chunk_buffer.push_back(resized_frame.clone());
            
            if (chunk_buffer.size() >= chunk_size) {
                // Process chunk
                try {
                    // Preprocess with compute backend if available
                    if (compute_backend_) {
                        std::vector<float> processed_data;
                        compute_backend_->process_frames_gpu(chunk_buffer, processed_data);
                    }
                    
                    auto result = ml_backend_.analyze(chunk_buffer, get_default_prompt());
                    callback(result);
                } catch (const std::exception& e) {
                    AnalysisResult error_result;
                    error_result.description = "Streaming analysis error: " + std::string(e.what());
                    callback(error_result);
                }
                
                chunk_buffer.clear();
            }
            
            frame_count++;
        }
        
        // Process remaining frames
        if (!chunk_buffer.empty()) {
            try {
                if (compute_backend_) {
                    std::vector<float> processed_data;
                    compute_backend_->process_frames_gpu(chunk_buffer, processed_data);
                }
                
                auto result = ml_backend_.analyze(chunk_buffer, get_default_prompt());
                callback(result);
            } catch (const std::exception& e) {
                AnalysisResult error_result;
                error_result.description = "Final chunk analysis error: " + std::string(e.what());
                callback(error_result);
            }
        }
        
        std::cout << "Streaming analysis completed. Processed " << frame_count << " frames." << std::endl;
    }

private:
    std::string get_default_prompt() const {
        return R"(
            Analyze this video sequence and provide:
            1. A detailed description of what's happening
            2. Key objects and people identified  
            3. Actions and activities taking place
            4. Scene context and environment
            5. Any notable events or changes over time
            
            Be specific and comprehensive in your analysis.
        )";
    }
    
    size_t calculate_optimal_batch_size(size_t total_videos) const {
        // Calculate based on available memory and thread count
        size_t memory_per_video = 256 * 1024 * 1024; // 256MB estimate per video
        size_t available_memory = config_.max_memory_mb * 1024 * 1024;
        size_t max_by_memory = std::max(1UL, available_memory / memory_per_video);
        
        size_t max_by_threads = static_cast<size_t>(config_.num_threads);
        
        return std::min({max_by_memory, max_by_threads, total_videos});
    }
    
    AnalysisConfig config_;
    FrameExtractor frame_extractor_;
    MLBackend ml_backend_;
    MemoryManager& memory_manager_;
    ComputeManager& compute_manager_;
    std::unique_ptr<ComputeBackend> compute_backend_;
};

// VideoProcessor public interface implementation
VideoProcessor::VideoProcessor(const AnalysisConfig& config) 
    : pimpl_(std::make_unique<Impl>(config)) {}

VideoProcessor::~VideoProcessor() = default;

VideoInfo VideoProcessor::get_video_info(const std::string& video_path) {
    return pimpl_->get_video_info(video_path);
}

std::vector<cv::Mat> VideoProcessor::extract_frames(const std::string& video_path) {
    return pimpl_->extract_frames(video_path);
}

AnalysisResult VideoProcessor::analyze_video(const std::string& video_path, 
                                           const std::string& custom_prompt) {
    return pimpl_->analyze_video(video_path, custom_prompt);
}

std::map<std::string, AnalysisResult> VideoProcessor::batch_analyze(
    const std::vector<std::string>& video_paths) {
    return pimpl_->batch_analyze(video_paths);
}

void VideoProcessor::start_stream_analysis(const std::string& video_path,
                                          std::function<void(const AnalysisResult&)> callback) {
    pimpl_->start_stream_analysis(video_path, callback);
}

} // namespace qwen 