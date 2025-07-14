#include "video_processor.hpp"
#include <iostream>
#include <chrono>
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] VIDEO_PATH\n"
              << "Options:\n"
              << "  -f, --frames NUM     Maximum frames to process (default: 16)\n"
              << "  -s, --strategy STR   Sampling strategy: uniform, keyframe (default: uniform)\n"
              << "  -t, --threads NUM    Number of threads (default: auto)\n"
              << "  -m, --memory NUM     Memory limit in MB (default: 8192)\n"
              << "  --cpu                Force CPU usage\n"
              << "  --prompt STR         Custom analysis prompt\n"
              << "  --output FILE        Output JSON file\n"
              << "  --batch              Process multiple videos\n"
              << "  --info               Show video information only\n"
              << "  -h, --help           Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    qwen::AnalysisConfig config;
    std::string video_path;
    std::string custom_prompt;
    std::string output_file;
    bool batch_mode = false;
    bool info_only = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-f" || arg == "--frames") {
            if (++i < argc) config.max_frames = std::stoi(argv[i]);
        } else if (arg == "-s" || arg == "--strategy") {
            if (++i < argc) config.sampling_strategy = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i < argc) config.num_threads = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--memory") {
            if (++i < argc) config.max_memory_mb = std::stoul(argv[i]);
        } else if (arg == "--cpu") {
            config.use_gpu = false;
        } else if (arg == "--prompt") {
            if (++i < argc) custom_prompt = argv[i];
        } else if (arg == "--output") {
            if (++i < argc) output_file = argv[i];
        } else if (arg == "--batch") {
            batch_mode = true;
        } else if (arg == "--info") {
            info_only = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (video_path.empty()) {
            video_path = arg;
        }
    }
    
    if (video_path.empty()) {
        std::cerr << "Error: No video path provided\n";
        return 1;
    }
    
    try {
        qwen::VideoProcessor processor(config);
        
        if (info_only) {
            // Just show video information
            auto info = processor.get_video_info(video_path);
            
            json info_json;
            info_json["video_path"] = video_path;
            info_json["total_frames"] = info.total_frames;
            info_json["fps"] = info.fps;
            info_json["duration"] = info.duration;
            info_json["frame_size"] = {info.frame_size.width, info.frame_size.height};
            info_json["codec"] = info.codec;
            
            std::cout << info_json.dump(2) << std::endl;
            return 0;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (batch_mode) {
            // Read video paths from file or directory
            std::vector<std::string> video_paths;
            
            // For now, just use the single video path as a test
            video_paths.push_back(video_path);
            
            auto results = processor.batch_analyze(video_paths);
            
            // Output batch results
            json output_json;
            for (const auto& [path, result] : results) {
                json result_json;
                result_json["description"] = result.description;
                result_json["objects"] = result.objects;
                result_json["actions"] = result.actions;
                result_json["scene_context"] = result.scene_context;
                result_json["temporal_events"] = result.temporal_events;
                result_json["confidence_score"] = result.confidence_score;
                result_json["processing_time_ms"] = result.processing_time.count();
                
                output_json[path] = result_json;
            }
            
            if (!output_file.empty()) {
                std::ofstream file(output_file);
                file << output_json.dump(2);
                std::cout << "Batch results saved to: " << output_file << std::endl;
            } else {
                std::cout << output_json.dump(2) << std::endl;
            }
            
        } else {
            // Single video analysis
            auto result = processor.analyze_video(video_path, custom_prompt);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
            
            // Create JSON output
            json output_json;
            output_json["video_path"] = video_path;
            output_json["description"] = result.description;
            output_json["objects"] = result.objects;
            output_json["actions"] = result.actions;
            output_json["scene_context"] = result.scene_context;
            output_json["temporal_events"] = result.temporal_events;
            output_json["confidence_score"] = result.confidence_score;
            output_json["processing_time_ms"] = result.processing_time.count();
            output_json["total_time_ms"] = total_time.count();
            
            if (!output_file.empty()) {
                std::ofstream file(output_file);
                file << output_json.dump(2);
                std::cout << "Results saved to: " << output_file << std::endl;
            } else {
                std::cout << output_json.dump(2) << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 