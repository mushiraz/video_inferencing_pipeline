#!/bin/bash

# Model Download Script for VCN Project
# Downloads all required AI models for video analysis

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

# Check if Ollama is running
check_ollama_service() {
    log_info "Checking Ollama service..."
    
    local max_attempts=5
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            log_success "Ollama service is running"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: Waiting for Ollama service..."
        sleep 3
        ((attempt++))
    done
    
    log_error "Ollama service is not running. Please start it first."
    return 1
}

# Get model size information
get_model_info() {
    local model_name="$1"
    
    case $model_name in
        "llava:7b")
            echo "4.7GB - Fast CPU-optimized vision model"
            ;;
        "llama3.2-vision:11b")
            echo "6.8GB - High-quality vision model"
            ;;
        "llama3.2:3b")
            echo "2.0GB - Fast text-only model"
            ;;
        "qwen2.5:7b")
            echo "4.4GB - Alternative text model"
            ;;
        *)
            echo "Unknown size - Custom model"
            ;;
    esac
}

# Download a single model
download_model() {
    local model_name="$1"
    local is_required="$2"
    local model_info=$(get_model_info "$model_name")
    
    log_info "Downloading model: $model_name ($model_info)"
    
    # Check if model already exists
    if ollama list | grep -q "^$model_name"; then
        log_info "Model $model_name already exists, skipping download"
        return 0
    fi
    
    # Download with progress
    local start_time=$(date +%s)
    
    if ollama pull "$model_name"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Model $model_name downloaded successfully (${duration}s)"
        return 0
    else
        if [[ "$is_required" == "true" ]]; then
            log_error "Failed to download required model: $model_name"
            return 1
        else
            log_warning "Failed to download optional model: $model_name"
            return 0
        fi
    fi
}

# Verify model functionality
verify_model() {
    local model_name="$1"
    
    log_info "Verifying model: $model_name"
    
    # Simple test prompt
    local test_prompt="Hello, can you respond with just 'OK'?"
    
    # Create test payload
    local payload=$(cat <<EOF
{
    "model": "$model_name",
    "prompt": "$test_prompt",
    "stream": false
}
EOF
)
    
    # Test the model
    local response=$(curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time 30)
    
    if echo "$response" | grep -q '"done":true'; then
        log_success "Model $model_name verified successfully"
        return 0
    else
        log_warning "Model $model_name verification failed"
        return 1
    fi
}

# Display storage requirements
show_storage_requirements() {
    log_info "Storage requirements for models:"
    echo "  Required models:"
    echo "    - llava:7b                 : 4.7GB (CPU-optimized vision)"
    echo "  Optional models:"
    echo "    - llama3.2-vision:11b      : 6.8GB (High-quality vision)"
    echo "    - llama3.2:3b              : 2.0GB (Fast text-only)"
    echo "    - qwen2.5:7b               : 4.4GB (Alternative text)"
    echo ""
    echo "  Total required space: ~4.7GB"
    echo "  Total with all optional: ~17.9GB"
    echo ""
}

# Check available disk space
check_disk_space() {
    log_info "Checking available disk space..."
    
    local available_space
    if command_exists df; then
        # Get available space in GB
        available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
        
        if [[ $available_space -lt 5 ]]; then
            log_error "Insufficient disk space. Available: ${available_space}GB, Required: 5GB minimum"
            return 1
        else
            log_success "Sufficient disk space available: ${available_space}GB"
            return 0
        fi
    else
        log_warning "Cannot check disk space (df command not available)"
        return 0
    fi
}

# Download required models
download_required_models() {
    log_info "Downloading required models..."
    
    local required_models=(
        "llava:7b"
    )
    
    for model in "${required_models[@]}"; do
        download_model "$model" "true" || return 1
    done
    
    log_success "All required models downloaded successfully"
}

# Download optional models
download_optional_models() {
    log_info "Downloading optional models..."
    
    local optional_models=(
        "llama3.2-vision:11b"
        "llama3.2:3b"
        "qwen2.5:7b"
    )
    
    for model in "${optional_models[@]}"; do
        download_model "$model" "false"
    done
    
    log_success "Optional models download completed"
}

# Verify all downloaded models
verify_all_models() {
    log_info "Verifying all downloaded models..."
    
    local models=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$")
    local verified_count=0
    local total_count=0
    
    while IFS= read -r model; do
        if [[ -n "$model" ]]; then
            ((total_count++))
            if verify_model "$model"; then
                ((verified_count++))
            fi
        fi
    done <<< "$models"
    
    log_info "Verified $verified_count/$total_count models"
    
    if [[ $verified_count -eq $total_count ]] && [[ $total_count -gt 0 ]]; then
        log_success "All models verified successfully"
        return 0
    else
        log_warning "Some models failed verification"
        return 1
    fi
}

# List downloaded models
list_models() {
    log_info "Currently downloaded models:"
    
    if command_exists ollama; then
        ollama list
    else
        log_error "Ollama not found"
        return 1
    fi
}

# Create model configuration file
create_model_config() {
    local config_file="models/model_config.json"
    
    log_info "Creating model configuration file..."
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Create configuration
    cat > "$config_file" << 'EOF'
{
    "models": {
        "vision": {
            "primary": "llava:7b",
            "fallback": "llama3.2-vision:11b",
            "description": "Vision-language models for image analysis"
        },
        "text": {
            "primary": "llama3.2:3b",
            "fallback": "qwen2.5:7b",
            "description": "Text-only models for summary generation"
        }
    },
    "endpoints": {
        "ollama": "http://localhost:11434",
        "api_version": "v1"
    },
    "settings": {
        "timeout": 300,
        "max_retries": 3,
        "stream": false
    }
}
EOF
    
    log_success "Model configuration created: $config_file"
}

# Main download function
main() {
    log_info "Starting model download for VCN Project..."
    
    # Check prerequisites
    if ! command_exists ollama; then
        log_error "Ollama not found. Please install Ollama first."
        exit 1
    fi
    
    # Check Ollama service
    check_ollama_service || exit 1
    
    # Show storage requirements
    show_storage_requirements
    
    # Check disk space
    check_disk_space || exit 1
    
    # Ask user for confirmation
    echo ""
    read -p "Do you want to download required models? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Download required models
        download_required_models || exit 1
        
        # Ask about optional models
        echo ""
        read -p "Do you want to download optional models? (y/N): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            download_optional_models
        fi
        
        # Verify models
        verify_all_models
        
        # Create configuration
        create_model_config
        
        # Show final status
        echo ""
        list_models
        
        log_success "Model download and setup completed successfully!"
        
        # Display usage information
        echo ""
        log_info "Usage examples:"
        echo "  # Test vision model:"
        echo "  echo 'Describe this image' | ollama run llava:7b"
        echo ""
        echo "  # Test text model:"
        echo "  echo 'Hello world' | ollama run llama3.2:3b"
        echo ""
        echo "  # Use in VCN project:"
        echo "  ./build/qwen_video_analyzer test_video.mp4"
        echo ""
    else
        log_info "Model download cancelled by user"
        exit 0
    fi
}

# Handle command line arguments
case "${1:-}" in
    "required")
        check_ollama_service || exit 1
        download_required_models || exit 1
        ;;
    "optional")
        check_ollama_service || exit 1
        download_optional_models
        ;;
    "verify")
        check_ollama_service || exit 1
        verify_all_models
        ;;
    "list")
        list_models
        ;;
    "config")
        create_model_config
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [required|optional|verify|list|config|help]"
        echo ""
        echo "Commands:"
        echo "  required  - Download only required models"
        echo "  optional  - Download only optional models"
        echo "  verify    - Verify all downloaded models"
        echo "  list      - List currently downloaded models"
        echo "  config    - Create model configuration file"
        echo "  help      - Show this help message"
        echo ""
        echo "If no command is specified, interactive mode is used."
        ;;
    *)
        main "$@"
        ;;
esac 