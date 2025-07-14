#include "video_processor.hpp"
#include "compute_manager.hpp"
#include "memory_manager.hpp"
#include <benchmark/benchmark.h>
#include <chrono>
#include <random>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace qwen {

class BenchmarkFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& /*state*/) override {
        config_.max_frames = 16;
        config_.target_size = cv::Size(512, 512);
        config_.use_gpu = true;
        config_.num_threads = std::thread::hardware_concurrency();
        
        // Create synthetic video data for testing
        create_synthetic_video();
        
        processor_ = std::make_unique<VideoProcessor>(config_);
    }
    
    void TearDown(const ::benchmark::State& /*state*/) override {
        processor_.reset();
        
        // Clean up test video
        std::remove("benchmark_video.mp4");
    }
    
protected:
    void create_synthetic_video() {
        // Create a temporary video file for benchmarking
        cv::VideoWriter writer;
        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        
        if (!writer.open("benchmark_video.mp4", fourcc, 30.0, cv::Size(1920, 1080))) {
            // Fallback to a different codec if H264 is not available
            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            writer.open("benchmark_video.mp4", fourcc, 30.0, cv::Size(1920, 1080));
        }
        
        if (!writer.isOpened()) {
            throw std::runtime_error("Failed to create benchmark video file");
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        
        // Create 300 frames (10 seconds at 30fps)
        for (int i = 0; i < 300; ++i) {
            cv::Mat frame = cv::Mat::zeros(1080, 1920, CV_8UC3);
            
            // Add some random content to make it more realistic
            for (int y = 0; y < frame.rows; y += 20) {
                for (int x = 0; x < frame.cols; x += 20) {
                    cv::Scalar color(dis(gen), dis(gen), dis(gen));
                    cv::rectangle(frame, 
                                cv::Point(x, y), 
                                cv::Point(x + 18, y + 18),
                                color,
                                -1);
                }
            }
            
            // Add some moving objects
            int circle_x = (i * 5) % frame.cols;
            int circle_y = 300 + static_cast<int>(100 * sin(i * 0.1));
            cv::circle(frame, cv::Point(circle_x, circle_y), 50, cv::Scalar(255, 255, 255), -1);
            
            writer << frame;
        }
        writer.release();
        
        std::cout << "Created benchmark video: benchmark_video.mp4" << std::endl;
    }
    
    AnalysisConfig config_;
    std::unique_ptr<VideoProcessor> processor_;
};

// Benchmark frame extraction performance
BENCHMARK_DEFINE_F(BenchmarkFixture, FrameExtraction)(benchmark::State& state) {
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Extract frames without ML analysis
        auto video_info = processor_->get_video_info("benchmark_video.mp4");
        auto frames = processor_->extract_frames("benchmark_video.mp4");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
        
        state.SetIterationTime(elapsed_seconds.count());
        state.counters["frames"] = static_cast<double>(frames.size());
        state.counters["fps"] = video_info.fps;
        state.counters["frames_per_second"] = static_cast<double>(frames.size()) / elapsed_seconds.count();
    }
}

// Benchmark full analysis pipeline
BENCHMARK_DEFINE_F(BenchmarkFixture, FullAnalysis)(benchmark::State& state) {
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = processor_->analyze_video("benchmark_video.mp4");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
        
        state.SetIterationTime(elapsed_seconds.count());
        state.counters["confidence"] = result.confidence_score;
        state.counters["processing_time_ms"] = static_cast<double>(result.processing_time.count());
    }
}

// Benchmark batch processing
BENCHMARK_DEFINE_F(BenchmarkFixture, BatchProcessing)(benchmark::State& state) {
    std::vector<std::string> video_paths;
    for (int i = 0; i < state.range(0); ++i) {
        video_paths.push_back("benchmark_video.mp4");
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto results = processor_->batch_analyze(video_paths);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
        
        state.SetIterationTime(elapsed_seconds.count());
        state.counters["videos"] = static_cast<double>(video_paths.size());
        state.counters["avg_time_per_video"] = elapsed_seconds.count() / video_paths.size();
        
        // Calculate average confidence
        double total_confidence = 0.0;
        for (const auto& [path, result] : results) {
            total_confidence += result.confidence_score;
        }
        state.counters["avg_confidence"] = total_confidence / results.size();
    }
}

// Benchmark compute backends
BENCHMARK_DEFINE_F(BenchmarkFixture, ComputeBackendPerformance)(benchmark::State& state) {
    // Create test frames
    std::vector<cv::Mat> test_frames;
    for (int i = 0; i < 16; ++i) {
        cv::Mat frame = cv::Mat::zeros(512, 512, CV_8UC3);
        cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        test_frames.push_back(frame);
    }
    
    // Get compute backend
    auto& compute_manager = ComputeManager::instance();
    auto backend = compute_manager.create_best_backend();
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<float> output;
        backend->process_frames_gpu(test_frames, output);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
        
        state.SetIterationTime(elapsed_seconds.count());
        state.counters["backend"] = 1; // Just to show which backend is being used
        state.counters["frames"] = static_cast<double>(test_frames.size());
        state.counters["output_size"] = static_cast<double>(output.size());
        state.counters["memory_usage_mb"] = static_cast<double>(backend->get_memory_usage()) / (1024 * 1024);
    }
    
    // Set label to show which backend was used
    state.SetLabel(backend->get_device_name());
}

// Benchmark memory management
BENCHMARK_DEFINE_F(BenchmarkFixture, MemoryManagement)(benchmark::State& state) {
    auto& memory_manager = MemoryManager::instance();
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simulate memory allocation and tracking
        size_t allocation_size = 100 * 1024 * 1024; // 100MB
        memory_manager.track_allocation("benchmark", allocation_size);
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        
        memory_manager.track_deallocation("benchmark", allocation_size);
        memory_manager.garbage_collect();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
        
        state.SetIterationTime(elapsed_seconds.count());
        state.counters["total_allocated"] = static_cast<double>(memory_manager.get_total_allocated());
        state.counters["peak_usage"] = static_cast<double>(memory_manager.get_peak_usage());
    }
}

// Benchmark different frame counts
BENCHMARK_DEFINE_F(BenchmarkFixture, FrameCountScaling)(benchmark::State& state) {
    // Update config for this benchmark
    config_.max_frames = static_cast<int>(state.range(0));
    processor_ = std::make_unique<VideoProcessor>(config_);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = processor_->analyze_video("benchmark_video.mp4");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
        
        state.SetIterationTime(elapsed_seconds.count());
        state.counters["max_frames"] = static_cast<double>(config_.max_frames);
        state.counters["confidence"] = result.confidence_score;
    }
}

// Register benchmarks
BENCHMARK_REGISTER_F(BenchmarkFixture, FrameExtraction)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BenchmarkFixture, FullAnalysis)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BenchmarkFixture, BatchProcessing)->Range(1, 8)->UseManualTime()->Unit(benchmark::kSecond);
BENCHMARK_REGISTER_F(BenchmarkFixture, ComputeBackendPerformance)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BenchmarkFixture, MemoryManagement)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BenchmarkFixture, FrameCountScaling)->Range(4, 32)->UseManualTime()->Unit(benchmark::kMillisecond);

} // namespace qwen

// Custom main function to add additional reporting
int main(int argc, char** argv) {
    std::cout << "Qwen Video Analysis Pipeline - Performance Benchmarks" << std::endl;
    std::cout << "=====================================================" << std::endl;
    
    // Print system information
    std::cout << "System Information:" << std::endl;
    std::cout << "  CPU Cores: " << std::thread::hardware_concurrency() << std::endl;
    
    // Print available compute backends
    auto& compute_manager = qwen::ComputeManager::instance();
    auto backends = compute_manager.get_available_backends();
    std::cout << "  Available Compute Backends: ";
    for (size_t i = 0; i < backends.size(); ++i) {
        std::cout << backends[i];
        if (i < backends.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Try to get the best backend and show its info
    try {
        auto backend = compute_manager.create_best_backend();
        std::cout << "  Primary Backend: " << backend->get_device_name() << std::endl;
        std::cout << "  Available Memory: " << (backend->get_memory_available() / (1024*1024*1024)) << "GB" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Backend Error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
    
    // Initialize and run benchmarks
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    
    std::cout << std::endl << "Benchmark completed!" << std::endl;
    
    return 0;
} 