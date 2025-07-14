/**
 * GPU-Accelerated Video Analysis C++ Integration
 * Uses rocDecode/rocJPEG for AMD GPU acceleration
 * Provides C++ wrapper for Python-based analysis pipeline
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <thread>

// JSON parsing (using nlohmann/json if available, simple parser fallback)
#ifdef USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#else
#include "simple_json_parser.h"
#endif

// ROCm/HIP includes for direct GPU operations (optional)
#ifdef USE_ROCM_DIRECT
#include <hip/hip_runtime.h>
#include <rocdecode.h>
#include <rocjpeg.h>
#endif

class GPUVideoAnalyzer {
private:
    std::string python_script_path;
    std::string ollama_url;
    bool gpu_acceleration_available;
    
public:
    struct AnalysisResult {
        bool success;
        std::string video_path;
        int frames_analyzed;
        int total_frames_extracted;
        double total_processing_time;
        std::string video_summary;
        std::string model_used;
        std::string acceleration_method;
        bool gpu_accelerated;
        std::vector<std::string> frame_descriptions;
        std::string error_message;
    };
    
    GPUVideoAnalyzer(const std::string& script_path = "gpu_accelerated_integration.py",
                     const std::string& url = "http://localhost:11434") 
        : python_script_path(script_path), ollama_url(url), gpu_acceleration_available(false) {
        
        // Check GPU availability
        checkGPUAvailability();
        
        std::cout << "🎮 GPU-Accelerated Video Analyzer Initialized\n";
        std::cout << "⚡ GPU Acceleration: " << (gpu_acceleration_available ? "Available" : "CPU Fallback") << "\n";
    }
    
    void checkGPUAvailability() {
        // Check if rocm-smi is available
        int result = system("rocm-smi > /dev/null 2>&1");
        gpu_acceleration_available = (result == 0);
        
        if (gpu_acceleration_available) {
            std::cout << "✅ AMD GPU detected with ROCm support\n";
        } else {
            std::cout << "⚠️  No AMD GPU detected - using CPU fallback\n";
        }
    }
    
    std::string getGPUStatus() {
        if (!gpu_acceleration_available) {
            return "CPU_FALLBACK";
        }
        
        // Get GPU utilization using rocm-smi
        std::string command = "rocm-smi --showuse --csv 2>/dev/null | tail -n +2 | cut -d',' -f2";
        FILE* pipe = popen(command.c_str(), "r");
        
        if (!pipe) {
            return "GPU_STATUS_UNKNOWN";
        }
        
        char buffer[128];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        pclose(pipe);
        
        // Parse utilization percentage
        if (!result.empty()) {
            try {
                int utilization = std::stoi(result);
                if (utilization < 50) return "GPU_READY";
                else if (utilization < 80) return "GPU_BUSY";
                else return "GPU_OVERLOADED";
            } catch (...) {
                return "GPU_STATUS_ERROR";
            }
        }
        
        return "GPU_UNKNOWN";
    }
    
    AnalysisResult analyzeVideo(const std::string& video_path, 
                               int max_frames = 10, 
                               bool output_json = true) {
        
        std::cout << "🎮 Starting GPU-Accelerated Video Analysis\n";
        std::cout << "📁 Video: " << video_path << "\n";
        std::cout << "🎯 Frames: " << max_frames << "\n";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Build Python command
        std::ostringstream cmd;
        cmd << "python3 " << python_script_path 
            << " \"" << video_path << "\"";
        
        if (output_json) {
            cmd << " --json";
        }
        
        cmd << " --frames " << max_frames;
        
        // Add GPU environment setup
        std::string full_command = "source gpu_env_setup.sh 2>/dev/null; " + cmd.str();
        
        std::cout << "🚀 Executing: " << cmd.str() << "\n";
        
        // Execute Python script and capture output
        FILE* pipe = popen(full_command.c_str(), "r");
        if (!pipe) {
            AnalysisResult error_result;
            error_result.success = false;
            error_result.error_message = "Failed to execute Python script";
            return error_result;
        }
        
        // Read output
        std::string output;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            output += buffer;
        }
        
        int exit_code = pclose(pipe);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "⏱️  Total execution time: " << duration.count() << "ms\n";
        
        if (exit_code != 0) {
            AnalysisResult error_result;
            error_result.success = false;
            error_result.error_message = "Python script failed with exit code: " + std::to_string(exit_code);
            error_result.total_processing_time = duration.count() / 1000.0;
            return error_result;
        }
        
        // Parse JSON result
        return parseAnalysisResult(output);
    }
    
    AnalysisResult parseAnalysisResult(const std::string& json_output) {
        AnalysisResult result;
        
        try {
#ifdef USE_NLOHMANN_JSON
            json parsed = json::parse(json_output);
            
            result.success = parsed.value("success", false);
            result.video_path = parsed.value("video_path", "");
            result.frames_analyzed = parsed.value("frames_analyzed", 0);
            result.total_frames_extracted = parsed.value("total_frames_extracted", 0);
            result.total_processing_time = parsed.value("total_processing_time", 0.0);
            result.video_summary = parsed.value("video_summary", "");
            result.model_used = parsed.value("model_used", "");
            result.acceleration_method = parsed.value("acceleration_method", "");
            result.gpu_accelerated = parsed.value("gpu_accelerated", false);
            result.error_message = parsed.value("error", "");
            
            // Extract frame descriptions
            if (parsed.contains("frame_details")) {
                for (const auto& frame : parsed["frame_details"]) {
                    if (frame.value("success", false)) {
                        result.frame_descriptions.push_back(frame.value("description", ""));
                    }
                }
            }
#else
            // Simple JSON parser fallback
            result = parseSimpleJSON(json_output);
#endif
            
        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = "JSON parsing error: " + std::string(e.what());
        }
        
        return result;
    }
    
    // Simple JSON parser for systems without nlohmann/json
    AnalysisResult parseSimpleJSON(const std::string& json_str) {
        AnalysisResult result;
        
        // Very basic JSON parsing - extract key values
        auto extractValue = [&](const std::string& key) -> std::string {
            std::string search = "\"" + key + "\": ";
            size_t pos = json_str.find(search);
            if (pos == std::string::npos) return "";
            
            pos += search.length();
            
            // Handle string values
            if (json_str[pos] == '"') {
                pos++;
                size_t end = json_str.find('"', pos);
                if (end != std::string::npos) {
                    return json_str.substr(pos, end - pos);
                }
            }
            // Handle numeric values
            else {
                size_t end = json_str.find_first_of(",}", pos);
                if (end != std::string::npos) {
                    return json_str.substr(pos, end - pos);
                }
            }
            
            return "";
        };
        
        result.success = (extractValue("success") == "true");
        result.video_path = extractValue("video_path");
        result.frames_analyzed = std::stoi(extractValue("frames_analyzed"));
        result.total_frames_extracted = std::stoi(extractValue("total_frames_extracted"));
        result.total_processing_time = std::stod(extractValue("total_processing_time"));
        result.video_summary = extractValue("video_summary");
        result.model_used = extractValue("model_used");
        result.acceleration_method = extractValue("acceleration_method");
        result.gpu_accelerated = (extractValue("gpu_accelerated") == "true");
        result.error_message = extractValue("error");
        
        return result;
    }
    
    void printResult(const AnalysisResult& result) {
        std::cout << "\n🎮 GPU-Accelerated Analysis Results\n";
        std::cout << "=" << std::string(50, '=') << "\n";
        
        if (result.success) {
            std::cout << "✅ Status: SUCCESS\n";
            std::cout << "📁 Video: " << result.video_path << "\n";
            std::cout << "🎯 Frames analyzed: " << result.frames_analyzed 
                      << "/" << result.total_frames_extracted << "\n";
            std::cout << "⏱️  Processing time: " << std::fixed << std::setprecision(2) 
                      << result.total_processing_time << "s\n";
            std::cout << "🎮 Acceleration: " << result.acceleration_method 
                      << (result.gpu_accelerated ? " (GPU)" : " (CPU)") << "\n";
            std::cout << "🤖 Model: " << result.model_used << "\n";
            
            std::cout << "\n📋 Video Summary:\n";
            std::cout << std::string(30, '-') << "\n";
            std::cout << result.video_summary << "\n";
            
            if (!result.frame_descriptions.empty()) {
                std::cout << "\n🎞️  Frame Details:\n";
                for (size_t i = 0; i < result.frame_descriptions.size(); ++i) {
                    std::cout << "Frame " << (i + 1) << ": " 
                              << result.frame_descriptions[i] << "\n";
                }
            }
        } else {
            std::cout << "❌ Status: FAILED\n";
            std::cout << "Error: " << result.error_message << "\n";
        }
    }
    
    // Batch processing for multiple videos
    std::vector<AnalysisResult> analyzeBatch(const std::vector<std::string>& video_paths,
                                           int max_frames = 5) {
        std::vector<AnalysisResult> results;
        
        std::cout << "🎮 Starting GPU Batch Analysis\n";
        std::cout << "📂 Processing " << video_paths.size() << " videos\n";
        
        for (size_t i = 0; i < video_paths.size(); ++i) {
            std::cout << "\n🎯 Processing " << (i + 1) << "/" << video_paths.size() 
                      << ": " << video_paths[i] << "\n";
            
            // Check GPU status before each video
            std::string gpu_status = getGPUStatus();
            std::cout << "🎮 GPU Status: " << gpu_status << "\n";
            
            // Add delay if GPU is busy
            if (gpu_status == "GPU_BUSY" || gpu_status == "GPU_OVERLOADED") {
                std::cout << "⏳ GPU busy, waiting 30s...\n";
                std::this_thread::sleep_for(std::chrono::seconds(30));
            }
            
            AnalysisResult result = analyzeVideo(video_paths[i], max_frames, true);
            results.push_back(result);
            
            if (result.success) {
                std::cout << "✅ Completed in " << result.total_processing_time << "s\n";
            } else {
                std::cout << "❌ Failed: " << result.error_message << "\n";
            }
        }
        
        return results;
    }
    
    void saveBatchReport(const std::vector<AnalysisResult>& results, 
                        const std::string& report_path = "gpu_batch_report.txt") {
        std::ofstream report(report_path);
        
        report << "GPU-Accelerated Video Analysis Batch Report\n";
        report << "============================================\n\n";
        
        int successful = 0;
        double total_time = 0.0;
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            
            report << "Video " << (i + 1) << ": " << result.video_path << "\n";
            report << "Status: " << (result.success ? "SUCCESS" : "FAILED") << "\n";
            
            if (result.success) {
                successful++;
                total_time += result.total_processing_time;
                
                report << "Frames: " << result.frames_analyzed << "/" 
                       << result.total_frames_extracted << "\n";
                report << "Time: " << result.total_processing_time << "s\n";
                report << "Method: " << result.acceleration_method << "\n";
                report << "Summary: " << result.video_summary << "\n";
            } else {
                report << "Error: " << result.error_message << "\n";
            }
            
            report << "\n" << std::string(50, '-') << "\n\n";
        }
        
        report << "SUMMARY\n";
        report << "=======\n";
        report << "Total videos: " << results.size() << "\n";
        report << "Successful: " << successful << "\n";
        report << "Failed: " << (results.size() - successful) << "\n";
        report << "Total processing time: " << total_time << "s\n";
        report << "Average per video: " << (successful > 0 ? total_time / successful : 0) << "s\n";
        
        std::cout << "📊 Batch report saved to: " << report_path << "\n";
    }
};

// Example usage and main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "🎮 GPU-Accelerated Video Analysis\n";
        std::cout << "Usage: " << argv[0] << " <video_path> [max_frames]\n";
        std::cout << "       " << argv[0] << " --batch <video1> <video2> ... [max_frames]\n";
        return 1;
    }
    
    GPUVideoAnalyzer analyzer;
    
    // Check for batch mode
    if (argc > 2 && std::string(argv[1]) == "--batch") {
        std::vector<std::string> video_paths;
        int max_frames = 5;  // Default for batch
        
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            
            // Check if it's a number (max_frames parameter)
            if (std::all_of(arg.begin(), arg.end(), ::isdigit)) {
                max_frames = std::stoi(arg);
            } else {
                video_paths.push_back(arg);
            }
        }
        
        if (video_paths.empty()) {
            std::cout << "❌ No video paths provided for batch processing\n";
            return 1;
        }
        
        auto results = analyzer.analyzeBatch(video_paths, max_frames);
        analyzer.saveBatchReport(results);
        
    } else {
        // Single video analysis
        std::string video_path = argv[1];
        int max_frames = (argc > 2) ? std::stoi(argv[2]) : 10;
        
        auto result = analyzer.analyzeVideo(video_path, max_frames, true);
        analyzer.printResult(result);
    }
    
    return 0;
} 