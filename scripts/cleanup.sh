#!/bin/bash

# Cleanup Script for VCN Project
# Removes downloaded models, temporary files, and build artifacts

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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Clean Ollama models
clean_ollama_models() {
    log_info "Cleaning Ollama models..."
    
    if ! command_exists ollama; then
        log_warning "Ollama not found, skipping model cleanup"
        return 0
    fi
    
    # Get list of models
    local models=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$")
    
    if [[ -z "$models" ]]; then
        log_info "No models to clean"
        return 0
    fi
    
    echo "Found models:"
    echo "$models"
    echo ""
    
    read -p "Do you want to remove all models? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        while IFS= read -r model; do
            if [[ -n "$model" ]]; then
                log_info "Removing model: $model"
                if ollama rm "$model"; then
                    log_success "Removed: $model"
                else
                    log_error "Failed to remove: $model"
                fi
            fi
        done <<< "$models"
    else
        log_info "Model cleanup cancelled by user"
    fi
}

# Clean temporary files
clean_temp_files() {
    log_info "Cleaning temporary files..."
    
    local temp_patterns=(
        "/tmp/ollama*"
        "/tmp/frame_*"
        "/tmp/test_*"
        "/tmp/concurrent_test_*"
        "/tmp/vcn_*"
    )
    
    for pattern in "${temp_patterns[@]}"; do
        if ls $pattern 2>/dev/null | head -1 | grep -q .; then
            log_info "Removing: $pattern"
            rm -f $pattern
        fi
    done
    
    log_success "Temporary files cleaned"
}

# Clean build artifacts
clean_build_artifacts() {
    log_info "Cleaning build artifacts..."
    
    local build_dirs=(
        "build"
        "CMakeFiles"
        ".cmake"
    )
    
    for dir in "${build_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            log_info "Removing build directory: $dir"
            rm -rf "$dir"
        fi
    done
    
    # Clean build files
    local build_files=(
        "CMakeCache.txt"
        "cmake_install.cmake"
        "Makefile"
        "CTestTestfile.cmake"
        "compile_commands.json"
    )
    
    for file in "${build_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_info "Removing build file: $file"
            rm -f "$file"
        fi
    done
    
    log_success "Build artifacts cleaned"
}

# Clean Python cache
clean_python_cache() {
    log_info "Cleaning Python cache..."
    
    # Remove __pycache__ directories
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    # Remove .pyc files
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove .pyo files
    find . -name "*.pyo" -delete 2>/dev/null || true
    
    log_success "Python cache cleaned"
}

# Clean log files
clean_log_files() {
    log_info "Cleaning log files..."
    
    local log_patterns=(
        "*.log"
        "logs/*.log"
        "/tmp/*.log"
    )
    
    for pattern in "${log_patterns[@]}"; do
        if ls $pattern 2>/dev/null | head -1 | grep -q .; then
            log_info "Removing logs: $pattern"
            rm -f $pattern
        fi
    done
    
    log_success "Log files cleaned"
}

# Clean downloaded models directory
clean_models_directory() {
    log_info "Cleaning models directory..."
    
    if [[ -d "models" ]]; then
        local size=$(du -sh models 2>/dev/null | cut -f1)
        echo "Models directory size: $size"
        echo ""
        
        read -p "Do you want to remove the models directory? (y/N): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf models
            log_success "Models directory removed"
        else
            log_info "Models directory cleanup cancelled"
        fi
    else
        log_info "No models directory found"
    fi
}

# Stop Ollama service
stop_ollama_service() {
    log_info "Stopping Ollama service..."
    
    if ! command_exists ollama; then
        log_warning "Ollama not found, skipping service stop"
        return 0
    fi
    
    # Try to stop via different methods
    if pgrep -f "ollama serve" > /dev/null; then
        log_info "Stopping Ollama server process..."
        pkill -f "ollama serve" || true
        sleep 2
        
        if pgrep -f "ollama serve" > /dev/null; then
            log_warning "Ollama server still running, trying force kill..."
            pkill -9 -f "ollama serve" || true
        fi
    fi
    
    # Stop systemd service if available
    if command_exists systemctl; then
        systemctl --user stop ollama 2>/dev/null || true
        sudo systemctl stop ollama 2>/dev/null || true
    fi
    
    # Stop Homebrew service on macOS
    if command_exists brew; then
        brew services stop ollama 2>/dev/null || true
    fi
    
    log_success "Ollama service stopped"
}

# Show cleanup summary
show_cleanup_summary() {
    log_info "Cleanup summary:"
    
    # Check disk space freed
    local current_space
    if command_exists df; then
        current_space=$(df -h . | tail -1 | awk '{print $4}')
        echo "  Available disk space: $current_space"
    fi
    
    # Check if Ollama is still running
    if pgrep -f "ollama serve" > /dev/null; then
        echo "  Ollama service: Still running"
    else
        echo "  Ollama service: Stopped"
    fi
    
    # Check remaining files
    local remaining_models=0
    if command_exists ollama; then
        remaining_models=$(ollama list | tail -n +2 | wc -l)
    fi
    echo "  Remaining models: $remaining_models"
    
    # Check build artifacts
    if [[ -d "build" ]]; then
        echo "  Build directory: Present"
    else
        echo "  Build directory: Cleaned"
    fi
    
    echo ""
    log_success "Cleanup completed!"
}

# Main cleanup function
main() {
    log_info "Starting cleanup for VCN Project..."
    echo ""
    
    # Ask for confirmation
    echo "This will clean up:"
    echo "  - Ollama models and service"
    echo "  - Build artifacts and cache"
    echo "  - Temporary files"
    echo "  - Log files"
    echo "  - Python cache"
    echo ""
    
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled by user"
        exit 0
    fi
    
    # Perform cleanup
    stop_ollama_service
    clean_ollama_models
    clean_temp_files
    clean_build_artifacts
    clean_python_cache
    clean_log_files
    clean_models_directory
    
    # Show summary
    show_cleanup_summary
}

# Handle command line arguments
case "${1:-}" in
    "models")
        clean_ollama_models
        ;;
    "build")
        clean_build_artifacts
        ;;
    "temp")
        clean_temp_files
        ;;
    "python")
        clean_python_cache
        ;;
    "logs")
        clean_log_files
        ;;
    "service")
        stop_ollama_service
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [models|build|temp|python|logs|service|help]"
        echo ""
        echo "Commands:"
        echo "  models   - Clean only Ollama models"
        echo "  build    - Clean only build artifacts"
        echo "  temp     - Clean only temporary files"
        echo "  python   - Clean only Python cache"
        echo "  logs     - Clean only log files"
        echo "  service  - Stop only Ollama service"
        echo "  help     - Show this help message"
        echo ""
        echo "If no command is specified, full cleanup is performed."
        ;;
    *)
        main "$@"
        ;;
esac 