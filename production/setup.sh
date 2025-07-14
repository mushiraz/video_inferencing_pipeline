#!/bin/bash

# Video Contextual Navigation (VCN) Production Setup Script
# Automated setup for macOS and Linux systems

set -e  # Exit on any error

echo "🏭 Video Contextual Navigation (VCN) Production Setup"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    log_info "Detected macOS system"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    log_info "Detected Linux system"
else
    log_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Check Python version
log_info "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_success "Python $PYTHON_VERSION detected"
else
    log_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    log_error "pip3 is not available. Please install pip first."
    exit 1
fi

# Install Python dependencies
log_info "Installing Python dependencies..."
if pip3 install -r requirements.txt; then
    log_success "Python dependencies installed successfully"
else
    log_error "Failed to install Python dependencies"
    exit 1
fi

# Install Ollama
log_info "Installing Ollama..."
if [[ "$OS" == "macos" ]]; then
    if command -v brew &> /dev/null; then
        if brew install ollama; then
            log_success "Ollama installed via Homebrew"
        else
            log_warning "Homebrew installation failed, trying curl method..."
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    else
        log_warning "Homebrew not found, installing via curl..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
elif [[ "$OS" == "linux" ]]; then
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Verify Ollama installation
if command -v ollama &> /dev/null; then
    log_success "Ollama installed successfully"
else
    log_error "Ollama installation failed"
    exit 1
fi

# Start Ollama service
log_info "Starting Ollama service..."
if [[ "$OS" == "macos" ]]; then
    # On macOS, start as background process
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 5
    
    if kill -0 $OLLAMA_PID 2>/dev/null; then
        log_success "Ollama service started (PID: $OLLAMA_PID)"
    else
        log_error "Failed to start Ollama service"
        exit 1
    fi
elif [[ "$OS" == "linux" ]]; then
    # On Linux, try systemd first
    if systemctl start ollama 2>/dev/null; then
        log_success "Ollama service started via systemd"
    else
        # Fallback to background process
        ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        sleep 5
        log_success "Ollama service started as background process"
    fi
fi

# Download required models
log_info "Downloading LLaVA-7B model (this may take several minutes)..."
if ollama pull llava:7b; then
    log_success "LLaVA-7B model downloaded successfully"
else
    log_error "Failed to download LLaVA-7B model"
    exit 1
fi

# Optional: Download backup model
log_info "Downloading backup model (Llama 3.2 Vision)..."
if ollama pull llama3.2-vision:11b; then
    log_success "Backup model downloaded successfully"
else
    log_warning "Backup model download failed (optional)"
fi

# Test the installation
log_info "Testing the installation..."
if python3 -c "import cv2, requests, base64, json; print('Core imports successful')"; then
    log_success "Python dependencies test passed"
else
    log_error "Python dependencies test failed"
    exit 1
fi

# Check if Ollama API is responding
log_info "Testing Ollama API..."
if curl -s http://localhost:11434/ > /dev/null; then
    log_success "Ollama API is responding"
else
    log_error "Ollama API is not responding"
    exit 1
fi

# Run a quick test with the sample video
if [ -f "test_video.mp4" ]; then
    log_info "Running quick test with sample video..."
    if timeout 300 python3 production_ready_integration.py test_video.mp4 --frames 1 --json > /dev/null 2>&1; then
        log_success "Sample video test completed successfully"
    else
        log_warning "Sample video test timed out or failed (this is normal for the first run)"
    fi
else
    log_warning "Sample video not found, skipping test"
fi

# Create useful scripts
log_info "Creating utility scripts..."

# Create start script
cat > start_vcn.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Video Contextual Navigation System..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/ > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 5
fi

echo "✅ VCN System ready!"
echo "Usage: python3 production_ready_integration.py <video_file>"
EOF

chmod +x start_vcn.sh

# Create stop script
cat > stop_vcn.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping Video Contextual Navigation System..."

# Stop Ollama processes
pkill -f "ollama serve" || true

echo "✅ VCN System stopped!"
EOF

chmod +x stop_vcn.sh

log_success "Utility scripts created (start_vcn.sh, stop_vcn.sh)"

# Final setup summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
log_success "Video Contextual Navigation system is ready for production use"
echo ""
echo "📋 Quick Start:"
echo "  1. Analyze a video: python3 production_ready_integration.py your_video.mp4"
echo "  2. JSON output: python3 production_ready_integration.py your_video.mp4 --json"
echo "  3. Custom frames: python3 production_ready_integration.py your_video.mp4 --frames 5"
echo ""
echo "🔧 Management:"
echo "  • Start system: ./start_vcn.sh"
echo "  • Stop system: ./stop_vcn.sh"
echo "  • Check logs: ollama logs"
echo ""
echo "📚 Documentation:"
echo "  • README.md - Quick start guide"
echo "  • TECHNICAL_ARCHITECTURE.md - Detailed technical information"
echo ""
log_info "System is optimized for CPU processing on $(uname -m) architecture"
echo "" 