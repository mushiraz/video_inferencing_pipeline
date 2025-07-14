/*
 * Video Contextual Navigation (VCN) - C++ Integration Example
 * 
 * This file demonstrates how to integrate the Python-based VCN pipeline
 * into a C++ application using subprocess calls and JSON parsing.
 */

#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>
#include <array>
#include <chrono>
#include <fstream>
#include <filesystem>

// Use nlohmann/json for JSON parsing
// Install: brew install nlohmann-json (macOS) or apt-get install nlohmann-json3-dev (Ubuntu)
#include <nlohmann/json.hpp>

namespace vcn {

/**
 * Result structure for video analysis
 */
struct VideoAnalysisResult {
    bool success;
    std::string video_path;
    int frames_analyzed;
    int total_frames_extracted;
    double total_processing_time;
    std::string video_summary;
    std::string model_used;
    std::string error_message;
    
    // Constructor for successful analysis
    VideoAnalysisResult(const std::string& path, int analyzed, int total, 
                       double time, const std::string& summary, const std::string& model)
        : success(true), video_path(path), frames_analyzed(analyzed), 
          total_frames_extracted(total), total_processing_time(time),
          video_summary(summary), model_used(model) {}
    
    // Constructor for failed analysis
    VideoAnalysisResult(const std::string& path, const std::string& error)
        : success(false), video_path(path), frames_analyzed(0), 
          total_frames_extracted(0), total_processing_time(0.0),
          error_message(error) {}
};

/**
 * VCN Pipeline Interface Class
 */
class VideoAnalyzer {
private:
    std::string python_script_path_;
    std::string python_executable_;
    int max_frames_;
    int timeout_seconds_;
    
    /**
     * Execute shell command and capture output
     */
    std::string execute_command(const std::string& command) {
        std::array<char, 128> buffer;
        std::string result;
        
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        
        return result;
    }
    
    /**
     * Check if Ollama service is running
     */
    bool is_ollama_running() {
        try {
            std::string result = execute_command("curl -s http://localhost:11434/ > /dev/null 2>&1; echo $?");
            return result.find("0") == 0;  // Exit code 0 means success
        } catch (...) {
            return false;
        }
    }
    
    /**
     * Start Ollama service if not running
     */
    bool ensure_ollama_running() {
        if (is_ollama_running()) {
            return true;
        }
        
        std::cout << "Starting Ollama service..." << std::endl;
        try {
            // Start Ollama in background
            execute_command("ollama serve > /dev/null 2>&1 &");
            
            // Wait for service to start
            for (int i = 0; i < 30; ++i) {  // Wait up to 30 seconds
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (is_ollama_running()) {
                    std::cout << "Ollama service started successfully." << std::endl;
                    return true;
                }
            }
        } catch (...) {
            return false;
        }
        
        return false;
    }
    
public:
    /**
     * Constructor
     */
    VideoAnalyzer(const std::string& script_path = "./production_ready_integration.py",
                  const std::string& python_exe = "python3",
                  int max_frames = 3,
                  int timeout = 600)
        : python_script_path_(script_path), python_executable_(python_exe),
          max_frames_(max_frames), timeout_seconds_(timeout) {
        
        // Verify Python script exists
        if (!std::filesystem::exists(python_script_path_)) {
            throw std::runtime_error("Python script not found: " + python_script_path_);
        }
    }
    
    /**
     * Analyze video and return structured result
     */
    VideoAnalysisResult analyze_video(const std::string& video_path) {
        // Verify video file exists
        if (!std::filesystem::exists(video_path)) {
            return VideoAnalysisResult(video_path, "Video file not found: " + video_path);
        }
        
        // Ensure Ollama is running
        if (!ensure_ollama_running()) {
            return VideoAnalysisResult(video_path, "Failed to start Ollama service");
        }
        
        try {
            // Build command
            std::string command = python_executable_ + " " + python_script_path_ + 
                                " \"" + video_path + "\" --json --frames " + 
                                std::to_string(max_frames_);
            
            std::cout << "Executing: " << command << std::endl;
            
            // Record start time
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Execute command
            std::string json_output = execute_command(command);
            
            // Record end time
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
            
            std::cout << "Analysis completed in " << duration.count() << " seconds" << std::endl;
            
            // Parse JSON response
            nlohmann::json result = nlohmann::json::parse(json_output);
            
            if (result["success"].get<bool>()) {
                return VideoAnalysisResult(
                    result["video_path"].get<std::string>(),
                    result["frames_analyzed"].get<int>(),
                    result["total_frames_extracted"].get<int>(),
                    result["total_processing_time"].get<double>(),
                    result["video_summary"].get<std::string>(),
                    result["model_used"].get<std::string>()
                );
            } else {
                std::string error = result.contains("error") ? 
                    result["error"].get<std::string>() : "Unknown error";
                return VideoAnalysisResult(video_path, error);
            }
            
        } catch (const nlohmann::json::parse_error& e) {
            return VideoAnalysisResult(video_path, "JSON parse error: " + std::string(e.what()));
        } catch (const std::exception& e) {
            return VideoAnalysisResult(video_path, "Analysis error: " + std::string(e.what()));
        }
    }
    
    /**
     * Batch analyze multiple videos
     */
    std::vector<VideoAnalysisResult> analyze_videos(const std::vector<std::string>& video_paths) {
        std::vector<VideoAnalysisResult> results;
        results.reserve(video_paths.size());
        
        for (const auto& path : video_paths) {
            std::cout << "Processing: " << path << std::endl;
            results.push_back(analyze_video(path));
        }
        
        return results;
    }
    
    /**
     * Get system status
     */
    bool is_system_ready() {
        return is_ollama_running() && std::filesystem::exists(python_script_path_);
    }
};

} // namespace vcn

/**
 * Example usage and testing
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_file> [video_file2] ..." << std::endl;
        std::cout << "Example: " << argv[0] << " test_video.mp4" << std::endl;
        return 1;
    }
    
    try {
        // Initialize analyzer
        vcn::VideoAnalyzer analyzer("./production_ready_integration.py", "python3", 3, 600);
        
        // Check system status
        if (!analyzer.is_system_ready()) {
            std::cerr << "System not ready. Please check Ollama installation and Python script." << std::endl;
            return 1;
        }
        
        std::cout << "Video Contextual Navigation - C++ Integration" << std::endl;
        std::cout << "=============================================" << std::endl;
        
        // Process all provided video files
        std::vector<std::string> video_paths;
        for (int i = 1; i < argc; ++i) {
            video_paths.push_back(argv[i]);
        }
        
        auto results = analyzer.analyze_videos(video_paths);
        
        // Display results
        std::cout << "\nAnalysis Results:" << std::endl;
        std::cout << "=================" << std::endl;
        
        for (const auto& result : results) {
            std::cout << "\nVideo: " << result.video_path << std::endl;
            if (result.success) {
                std::cout << "✅ Status: SUCCESS" << std::endl;
                std::cout << "📊 Frames analyzed: " << result.frames_analyzed << "/" 
                         << result.total_frames_extracted << std::endl;
                std::cout << "⏱️  Processing time: " << result.total_processing_time << "s" << std::endl;
                std::cout << "🤖 Model: " << result.model_used << std::endl;
                std::cout << "📝 Summary: " << result.video_summary << std::endl;
            } else {
                std::cout << "❌ Status: FAILED" << std::endl;
                std::cout << "💥 Error: " << result.error_message << std::endl;
            }
            std::cout << std::string(50, '-') << std::endl;
        }
        
        // Summary statistics
        int successful = 0;
        for (const auto& result : results) {
            if (result.success) successful++;
        }
        
        std::cout << "\nSummary: " << successful << "/" << results.size() 
                  << " videos processed successfully" << std::endl;
        
        return successful == results.size() ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

/* 
 * Compilation instructions:
 * 
 * macOS (with Homebrew):
 * g++ -std=c++17 -I/opt/homebrew/include cpp_integration_example.cpp -o vcn_analyzer
 * 
 * Linux (Ubuntu):
 * g++ -std=c++17 -I/usr/include/nlohmann cpp_integration_example.cpp -o vcn_analyzer
 * 
 * Usage:
 * ./vcn_analyzer test_video.mp4
 * ./vcn_analyzer video1.mp4 video2.mp4 video3.mp4
 */ 