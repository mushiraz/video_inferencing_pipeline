# Video Contextual Navigation (VCN) - Production Deployment Guide

This guide provides step-by-step instructions for deploying the VCN production system in various environments.

## 🚀 Quick Deployment (5 minutes)

### Automated Setup
```bash
# Clone or extract the production files
cd production/

# Run automated setup (installs everything)
chmod +x setup.sh
./setup.sh

# Test the system
python3 test_production.py
```

### Manual Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install and start Ollama
brew install ollama  # macOS
# OR: curl -fsSL https://ollama.ai/install.sh | sh  # Linux
ollama serve &

# 3. Download required models
ollama pull llava:7b

# 4. Test the system
python3 production_ready_integration.py test_video.mp4 --json
```

## 📋 Production Files Overview

### Core System Files
- **`production_ready_integration.py`** - Main video analysis pipeline
- **`requirements.txt`** - Python dependencies
- **`setup.sh`** - Automated setup script
- **`test_production.py`** - System validation script

### Integration Files  
- **`cpp_integration_example.cpp`** - C++ integration example
- **`ollama_bridge.py`** - Direct Ollama API utility
- **`Makefile`** - C++ build automation

### Documentation
- **`README.md`** - Quick start guide
- **`TECHNICAL_ARCHITECTURE.md`** - Detailed technical documentation
- **`DEPLOYMENT_GUIDE.md`** - This deployment guide

### Support Files
- **`test_video.mp4`** - Sample video for testing
- **`.gitignore`** - Git ignore patterns

## 🏗️ System Requirements

### Minimum Requirements
- **OS**: macOS 11+ or Ubuntu 20.04+
- **CPU**: 4 cores, 2.5GHz+ (optimized for CPU processing)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Python**: 3.8 or higher

### Recommended Specifications
- **CPU**: 8+ cores (Apple M1/M2 or Intel/AMD equivalent)
- **RAM**: 16GB+ for optimal performance
- **Storage**: SSD for temporary file operations

## 🔧 Deployment Scenarios

### 1. Development Environment

```bash
# Quick setup for development/testing
cd production/
pip install -r requirements.txt
ollama serve &
ollama pull llava:7b
python3 test_production.py
```

### 2. Production Server (Linux)

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3 python3-pip curl

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Setup VCN system
cd production/
pip3 install -r requirements.txt
sudo systemctl start ollama  # or ./setup.sh
ollama pull llava:7b

# Optional: Install as system service
sudo cp production_ready_integration.py /usr/local/bin/vcn-analyze
```

### 3. Container Deployment (Docker)

```dockerfile
# Example Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy VCN files
COPY production/ /app/
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose Ollama port
EXPOSE 11434

# Start script
CMD ["./setup.sh"]
```

### 4. C++ Application Integration

```bash
# Install C++ dependencies
make deps

# Build C++ integration
make all

# Test C++ integration
./vcn_analyzer test_video.mp4
```

## 🎯 Performance Tuning

### CPU Optimization
```bash
# Set environment variables for better performance
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_GPU_OVERHEAD=0
```

### Memory Management
```bash
# For systems with limited RAM
export OLLAMA_KEEP_ALIVE=5m  # Unload model after 5 minutes
```

### Processing Parameters
```python
# In production_ready_integration.py, adjust:
max_frames = 3          # Fewer frames for speed
timeout = 180           # Shorter timeout
max_size = 256          # Smaller images for faster processing
```

## 🔍 Monitoring & Maintenance

### Health Checks
```bash
# Check system status
python3 test_production.py

# Check Ollama service
curl http://localhost:11434/

# Monitor resource usage
htop  # or Activity Monitor on macOS
```

### Log Management
```bash
# Ollama logs
ollama logs

# System logs (Linux)
journalctl -u ollama

# Custom logging
export VCN_LOG_LEVEL=DEBUG
python3 production_ready_integration.py video.mp4 > analysis.log 2>&1
```

### Model Updates
```bash
# Update models
ollama pull llava:7b
ollama pull llama3.2-vision:11b

# List available models
ollama list
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Ollama Service Won't Start
```bash
# Check if port is in use
lsof -i :11434

# Kill existing processes
pkill -f ollama

# Restart service
ollama serve &
```

#### 2. Model Download Fails
```bash
# Check internet connection
ping registry.ollama.ai

# Try different model source
ollama pull llava:7b --insecure

# Manual download
curl -O https://registry.ollama.ai/v2/library/llava/blobs/sha256-...
```

#### 3. High CPU Usage
```bash
# Reduce concurrent processing
export OLLAMA_NUM_PARALLEL=1

# Use smaller models
# Change preferred_model in production_ready_integration.py
```

#### 4. Memory Issues
```bash
# Clear model cache
ollama stop
rm -rf ~/.ollama/models/blobs/
ollama serve &
ollama pull llava:7b
```

### Performance Issues

#### Slow Processing
- **Reduce frame count**: Use `--frames 3` instead of 10
- **Smaller images**: Modify `max_size` in frame extraction
- **Faster model**: Ensure using `llava:7b` not `llama3.2-vision:11b`

#### System Overload
- **Monitor CPU**: Keep below 80% average
- **Check memory**: Ensure 4GB+ available
- **Restart Ollama**: Automatic restart included in pipeline

## 📊 Production Metrics

### Expected Performance (MacBook Pro M2)
- **Frame Extraction**: 1-2 seconds
- **Single Frame Analysis**: 2-3 minutes
- **Summary Generation**: 30-60 seconds
- **Total (3 frames)**: 8-10 minutes

### Resource Usage
- **CPU**: 85-100% during analysis
- **Memory**: 2-4GB for model
- **Disk**: <100MB temporary files
- **Network**: Local only (no external calls)

## 🔐 Security Considerations

### Data Privacy
- All processing is local (no cloud services)
- Temporary files automatically cleaned
- No persistent storage of video content

### Access Control
```bash
# Restrict file permissions
chmod 750 production_ready_integration.py
chown root:vcn-users production/

# Firewall (if needed)
sudo ufw deny 11434  # Block external Ollama access
```

### Production Hardening
```bash
# Run as non-root user
useradd -m vcn-user
sudo -u vcn-user python3 production_ready_integration.py video.mp4

# Limit resource usage
ulimit -m 4194304  # 4GB memory limit
ulimit -t 600      # 10 minute CPU time limit
```

## 📈 Scaling Options

### Horizontal Scaling
```bash
# Multiple instances with load balancer
# Instance 1: Port 11434
# Instance 2: Port 11435
# Instance 3: Port 11436

export OLLAMA_PORT=11435
ollama serve --port 11435 &
```

### Queue Management
```python
# Example task queue integration
import celery

@celery.task
def analyze_video_task(video_path):
    analyzer = ProductionVideoAnalyzer()
    return analyzer.analyze_video_production(video_path)
```

### Cloud Deployment
```yaml
# Kubernetes example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vcn-analyzer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vcn-analyzer
  template:
    metadata:
      labels:
        app: vcn-analyzer
    spec:
      containers:
      - name: vcn
        image: vcn-analyzer:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

## 📞 Support & Contact

### Getting Help
1. Check this deployment guide
2. Review `README.md` for quick start
3. Run `python3 test_production.py` for diagnostics
4. Check `TECHNICAL_ARCHITECTURE.md` for details

### Common Commands
```bash
# System check
make check

# Full setup
./setup.sh

# Test system
python3 test_production.py

# Build C++ integration
make all

# Clean build
make clean
```

---

**Production Deployment Complete!** 🎉

The VCN system is now ready for production use with comprehensive monitoring, error handling, and integration capabilities. 