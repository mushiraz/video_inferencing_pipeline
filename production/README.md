# Video Contextual Navigation (VCN) - Production System

A production-ready video analysis pipeline that transforms video content into intelligent text summaries using LLaVA-7B vision model through Ollama.

## 🏗️ System Overview

This system processes video files by extracting key frames, analyzing them with a vision-language model, and generating cohesive summaries. It's optimized for CPU-only processing on macOS systems.

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install and start Ollama
brew install ollama
ollama serve &

# Pull the required model
ollama pull llava:7b
```

### 2. Run Video Analysis

```bash
# Basic analysis
python3 production_ready_integration.py test_video.mp4

# JSON output for integration
python3 production_ready_integration.py test_video.mp4 --json

# Custom number of frames
python3 production_ready_integration.py test_video.mp4 --frames 5
```

## 📁 Production Files

### Core Components

- **`production_ready_integration.py`** - Main video analysis pipeline
- **`ollama_bridge.py`** - Utility for direct Ollama API communication
- **`requirements.txt`** - Python dependencies
- **`test_video.mp4`** - Sample video for testing
- **`TECHNICAL_ARCHITECTURE.md`** - Detailed technical documentation

### Key Features

- **CPU-Optimized**: Designed for high-performance CPU processing
- **System Monitoring**: Automatic load detection and Ollama restart
- **Robust Error Handling**: Fallback mechanisms and timeout management
- **Production Ready**: JSON output for easy C++ integration
- **Scalable**: Processes 3-10 frames for optimal speed/quality balance

## 🔧 Configuration

### Model Settings
- **Default Model**: `llava:7b` (CPU-optimized)
- **Fallback Model**: `llama3.2-vision:11b` (higher quality)
- **Frame Resolution**: 384px maximum dimension
- **JPEG Quality**: 70% for optimal speed/quality

### Performance Parameters
- **Timeout**: 300 seconds per frame analysis
- **Max Frames**: 10 frames per video (configurable)
- **Load Threshold**: System load < 8.0 for processing
- **Memory Management**: Automatic cleanup of temporary files

## 📊 Expected Performance

### Typical Processing Times (MacBook Pro M2)
- **Frame Extraction**: ~1-2 seconds
- **Single Frame Analysis**: ~2-3 minutes  
- **Summary Generation**: ~1 minute
- **Total for 3 frames**: ~8-10 minutes

### Resource Usage
- **CPU**: 85-100% during analysis
- **Memory**: ~2-4GB for model loading
- **Disk**: Minimal (temporary frame files)

## 🔗 Integration

### C++ Integration Example

```cpp
#include <cstdlib>
#include <string>
#include <nlohmann/json.hpp>

std::string analyze_video(const std::string& video_path) {
    std::string command = "python3 production_ready_integration.py " + 
                         video_path + " --json";
    
    FILE* pipe = popen(command.c_str(), "r");
    // Read JSON response and parse with nlohmann::json
    
    return result;
}
```

### JSON Output Format

```json
{
  "success": true,
  "video_path": "test_video.mp4",
  "frames_analyzed": 3,
  "total_frames_extracted": 3,
  "total_processing_time": 487.25,
  "video_summary": "The video shows a person demonstrating...",
  "model_used": "llava:7b"
}
```

## 🐛 Troubleshooting

### Common Issues

1. **High CPU Load**
   - System automatically restarts Ollama with optimizations
   - Reduces processing to single-threaded mode

2. **Model Loading Failures**
   - Ensure Ollama is running: `ollama serve`
   - Verify model is pulled: `ollama pull llava:7b`

3. **Memory Issues**
   - Pipeline automatically cleans temporary files
   - Monitor with `htop` or Activity Monitor

### Debug Mode

```bash
# Enable verbose logging
export OLLAMA_DEBUG=1
python3 production_ready_integration.py test_video.mp4
```

## 📈 Performance Optimization

### For Better Performance
- Close CPU-intensive applications
- Use SSD storage for temporary files
- Consider cloud-based processing for large batches

### Scaling Options
- **Horizontal**: Multiple instances with load balancing
- **Vertical**: More CPU cores and RAM
- **GPU**: Future integration with CUDA/Metal acceleration

## 🔐 Production Considerations

### Security
- Temporary files are automatically cleaned up
- No persistent storage of sensitive data
- Local processing (no data sent to external services)

### Monitoring
- Built-in system load monitoring
- Processing time tracking
- Success/failure rate logging

### Maintenance
- Regular Ollama model updates
- System resource monitoring
- Log rotation for long-running deployments

## 📋 Dependencies

### Required
- Python 3.8+
- OpenCV 4.8+
- Ollama with LLaVA-7B model
- macOS (optimized) or Linux

### Optional
- psutil for system monitoring
- tqdm for progress bars
- pytest for testing

## 🆘 Support

For technical issues or questions:
1. Check the `TECHNICAL_ARCHITECTURE.md` for detailed information
2. Verify system requirements and dependencies
3. Monitor Ollama logs: `ollama logs`
4. Check system resources with Activity Monitor

## 📄 License

Production-ready implementation of Video Contextual Navigation system. 