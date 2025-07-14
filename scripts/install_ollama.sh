#!/bin/bash

# Ollama Installation Script for VCN Project
# Supports macOS, Linux, and Windows (WSL)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Ollama on macOS
install_ollama_macos() {
    log_info "Installing Ollama on macOS..."
    
    # Try Homebrew first
    if command_exists brew; then
        log_info "Using Homebrew to install Ollama..."
        if brew install ollama; then
            log_success "Ollama installed via Homebrew"
            return 0
        else
            log_warning "Homebrew installation failed, trying curl method..."
        fi
    else
        log_warning "Homebrew not found, using curl method..."
    fi
    
    # Fallback to curl installation
    log_info "Installing Ollama via curl..."
    if curl -fsSL https://ollama.ai/install.sh | sh; then
        log_success "Ollama installed via curl"
        return 0
    else
        log_error "Failed to install Ollama via curl"
        return 1
    fi
}

# Install Ollama on Linux
install_ollama_linux() {
    log_info "Installing Ollama on Linux..."
    
    # Check if we're in a container or have sudo access
    if [[ -f /.dockerenv ]] || [[ "$EUID" -eq 0 ]]; then
        log_info "Running in container or as root, installing directly..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        log_info "Installing Ollama for current user..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
    # Try to enable systemd service if available
    if command_exists systemctl && [[ "$EUID" -ne 0 ]]; then
        log_info "Attempting to enable Ollama systemd service..."
        if systemctl --user enable ollama 2>/dev/null; then
            log_success "Ollama systemd service enabled"
        else
            log_warning "Could not enable systemd service, will run manually"
        fi
    fi
}

# Install Ollama on Windows (WSL)
install_ollama_windows() {
    log_info "Installing Ollama on Windows (WSL)..."
    curl -fsSL https://ollama.ai/install.sh | sh
}

# Verify Ollama installation
verify_ollama() {
    log_info "Verifying Ollama installation..."
    
    if command_exists ollama; then
        local version=$(ollama --version 2>/dev/null | head -n1)
        log_success "Ollama installed successfully: $version"
        return 0
    else
        log_error "Ollama installation verification failed"
        return 1
    fi
}

# Start Ollama service
start_ollama() {
    log_info "Starting Ollama service..."
    
    local os=$(detect_os)
    
    case $os in
        "macos")
            # On macOS, start as background process
            if pgrep -f "ollama serve" > /dev/null; then
                log_info "Ollama service already running"
                return 0
            fi
            
            log_info "Starting Ollama server in background..."
            nohup ollama serve > /tmp/ollama.log 2>&1 &
            local pid=$!
            
            # Wait a bit and check if it's running
            sleep 5
            if kill -0 $pid 2>/dev/null; then
                log_success "Ollama service started (PID: $pid)"
                echo $pid > /tmp/ollama.pid
                return 0
            else
                log_error "Failed to start Ollama service"
                return 1
            fi
            ;;
            
        "linux")
            # Try systemd first
            if command_exists systemctl; then
                if systemctl --user start ollama 2>/dev/null; then
                    log_success "Ollama service started via systemd"
                    return 0
                elif sudo systemctl start ollama 2>/dev/null; then
                    log_success "Ollama service started via system systemd"
                    return 0
                fi
            fi
            
            # Fallback to background process
            if pgrep -f "ollama serve" > /dev/null; then
                log_info "Ollama service already running"
                return 0
            fi
            
            log_info "Starting Ollama server in background..."
            nohup ollama serve > /tmp/ollama.log 2>&1 &
            local pid=$!
            
            sleep 5
            if kill -0 $pid 2>/dev/null; then
                log_success "Ollama service started (PID: $pid)"
                echo $pid > /tmp/ollama.pid
                return 0
            else
                log_error "Failed to start Ollama service"
                return 1
            fi
            ;;
            
        "windows")
            # Windows/WSL - background process
            if pgrep -f "ollama serve" > /dev/null; then
                log_info "Ollama service already running"
                return 0
            fi
            
            log_info "Starting Ollama server in background..."
            nohup ollama serve > /tmp/ollama.log 2>&1 &
            local pid=$!
            
            sleep 5
            if kill -0 $pid 2>/dev/null; then
                log_success "Ollama service started (PID: $pid)"
                echo $pid > /tmp/ollama.pid
                return 0
            else
                log_error "Failed to start Ollama service"
                return 1
            fi
            ;;
    esac
}

# Check if Ollama service is responsive
check_ollama_health() {
    log_info "Checking Ollama service health..."
    
    local max_attempts=10
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            log_success "Ollama service is responsive"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: Waiting for Ollama service..."
        sleep 2
        ((attempt++))
    done
    
    log_error "Ollama service is not responsive after $max_attempts attempts"
    return 1
}

# Setup Ollama environment variables
setup_ollama_env() {
    log_info "Setting up Ollama environment variables..."
    
    # Create environment file
    cat > /tmp/ollama_env.sh << 'EOF'
# Ollama Environment Configuration for VCN Project
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_ORIGINS="*"
export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_MAX_LOADED_MODELS="2"
export OLLAMA_KEEP_ALIVE="10m"
export OLLAMA_LOAD_TIMEOUT="300s"
export OLLAMA_GPU_OVERHEAD="0"
export OLLAMA_DEBUG="false"
EOF
    
    # Source the environment file
    source /tmp/ollama_env.sh
    
    log_success "Ollama environment configured"
}

# Main installation function
main() {
    log_info "Starting Ollama installation for VCN Project..."
    
    local os=$(detect_os)
    log_info "Detected OS: $os"
    
    # Check if Ollama is already installed
    if command_exists ollama; then
        log_info "Ollama is already installed"
        if verify_ollama; then
            log_info "Proceeding to start service..."
        else
            log_error "Existing Ollama installation is corrupted"
            exit 1
        fi
    else
        # Install Ollama based on OS
        case $os in
            "macos")
                install_ollama_macos || exit 1
                ;;
            "linux")
                install_ollama_linux || exit 1
                ;;
            "windows")
                install_ollama_windows || exit 1
                ;;
            *)
                log_error "Unsupported operating system: $os"
                exit 1
                ;;
        esac
        
        # Verify installation
        verify_ollama || exit 1
    fi
    
    # Setup environment
    setup_ollama_env
    
    # Start Ollama service
    start_ollama || exit 1
    
    # Check service health
    check_ollama_health || exit 1
    
    log_success "Ollama installation and setup completed successfully!"
    log_info "Ollama is running at: http://localhost:11434"
    log_info "Log file: /tmp/ollama.log"
    
    # Display next steps
    echo ""
    log_info "Next steps:"
    echo "  1. Download models: cmake --build build --target download_models"
    echo "  2. Test installation: ollama list"
    echo "  3. Run the application: ./build/qwen_video_analyzer"
    echo ""
}

# Run main function
main "$@" 