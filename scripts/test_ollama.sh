#!/bin/bash

# Ollama Integration Test Script for VCN Project
# Tests Ollama service and model functionality

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

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Test function wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    log_info "Running test: $test_name"
    
    if $test_function; then
        log_success "✓ $test_name"
        ((TESTS_PASSED++))
        return 0
    else
        log_error "✗ $test_name"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test 1: Check if Ollama is installed
test_ollama_installation() {
    command -v ollama >/dev/null 2>&1
}

# Test 2: Check if Ollama service is running
test_ollama_service() {
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1
}

# Test 3: Check if required models are available
test_required_models() {
    local required_models=("llava:7b")
    
    for model in "${required_models[@]}"; do
        if ! ollama list | grep -q "^$model"; then
            log_error "Required model not found: $model"
            return 1
        fi
    done
    
    return 0
}

# Test 4: Test vision model functionality
test_vision_model() {
    local model="llava:7b"
    
    # Check if model exists
    if ! ollama list | grep -q "^$model"; then
        log_warning "Vision model not available: $model"
        return 1
    fi
    
    # Create test payload
    local payload=$(cat <<EOF
{
    "model": "$model",
    "prompt": "Respond with exactly 'TEST_PASSED' and nothing else.",
    "stream": false
}
EOF
)
    
    # Test the model
    local response=$(curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time 60)
    
    if echo "$response" | grep -q '"done":true'; then
        return 0
    else
        log_error "Vision model test failed: $response"
        return 1
    fi
}

# Test 5: Test text model functionality (if available)
test_text_model() {
    local model="llama3.2:3b"
    
    # Check if model exists
    if ! ollama list | grep -q "^$model"; then
        log_warning "Text model not available: $model (optional)"
        return 0  # This is optional, so don't fail
    fi
    
    # Create test payload
    local payload=$(cat <<EOF
{
    "model": "$model",
    "prompt": "Say 'OK' and nothing else.",
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
        return 0
    else
        log_warning "Text model test failed: $response"
        return 1
    fi
}

# Test 6: Test Python bridge functionality
test_python_bridge() {
    local bridge_script="../production/ollama_bridge.py"
    
    if [[ ! -f "$bridge_script" ]]; then
        log_warning "Python bridge script not found: $bridge_script"
        return 1
    fi
    
    # Create a test image (simple 1x1 pixel)
    local test_image="/tmp/test_image.jpg"
    python3 -c "
from PIL import Image
img = Image.new('RGB', (1, 1), color='red')
img.save('$test_image')
"
    
    if [[ ! -f "$test_image" ]]; then
        log_error "Could not create test image"
        return 1
    fi
    
    # Test the bridge
    local result=$(python3 "$bridge_script" "What color is this?" "$test_image" 2>/dev/null)
    
    # Clean up
    rm -f "$test_image"
    
    if echo "$result" | grep -q '"success":true'; then
        return 0
    else
        log_error "Python bridge test failed: $result"
        return 1
    fi
}

# Test 7: Test model configuration
test_model_config() {
    local config_file="../models/model_config.json"
    
    if [[ ! -f "$config_file" ]]; then
        log_warning "Model configuration file not found: $config_file"
        return 1
    fi
    
    # Validate JSON
    if python3 -c "import json; json.load(open('$config_file'))" 2>/dev/null; then
        return 0
    else
        log_error "Invalid JSON in model configuration"
        return 1
    fi
}

# Test 8: Test API endpoints
test_api_endpoints() {
    local endpoints=(
        "http://localhost:11434/api/tags"
        "http://localhost:11434/api/version"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if ! curl -s "$endpoint" >/dev/null 2>&1; then
            log_error "API endpoint not responding: $endpoint"
            return 1
        fi
    done
    
    return 0
}

# Test 9: Test concurrent requests
test_concurrent_requests() {
    local model="llava:7b"
    
    if ! ollama list | grep -q "^$model"; then
        log_warning "Vision model not available for concurrent test"
        return 1
    fi
    
    # Create test payload
    local payload=$(cat <<EOF
{
    "model": "$model",
    "prompt": "Say 'CONCURRENT_TEST' and nothing else.",
    "stream": false
}
EOF
)
    
    # Run 3 concurrent requests
    local pids=()
    for i in {1..3}; do
        curl -s -X POST http://localhost:11434/api/generate \
            -H "Content-Type: application/json" \
            -d "$payload" \
            --max-time 30 > "/tmp/concurrent_test_$i.json" &
        pids+=($!)
    done
    
    # Wait for all requests to complete
    local success_count=0
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            ((success_count++))
        fi
    done
    
    # Check results
    for i in {1..3}; do
        if [[ -f "/tmp/concurrent_test_$i.json" ]]; then
            if grep -q '"done":true' "/tmp/concurrent_test_$i.json"; then
                ((success_count++))
            fi
            rm -f "/tmp/concurrent_test_$i.json"
        fi
    done
    
    if [[ $success_count -ge 2 ]]; then
        return 0
    else
        log_error "Concurrent requests test failed: only $success_count/3 succeeded"
        return 1
    fi
}

# Test 10: Performance benchmark
test_performance() {
    local model="llava:7b"
    
    if ! ollama list | grep -q "^$model"; then
        log_warning "Vision model not available for performance test"
        return 1
    fi
    
    log_info "Running performance benchmark..."
    
    # Create test payload
    local payload=$(cat <<EOF
{
    "model": "$model",
    "prompt": "Hello",
    "stream": false
}
EOF
)
    
    # Measure response time
    local start_time=$(date +%s.%N)
    local response=$(curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time 60)
    local end_time=$(date +%s.%N)
    
    local duration=$(echo "$end_time - $start_time" | bc)
    
    if echo "$response" | grep -q '"done":true'; then
        log_info "Response time: ${duration}s"
        
        # Check if response time is reasonable (< 30 seconds)
        if (( $(echo "$duration < 30" | bc -l) )); then
            return 0
        else
            log_warning "Response time too slow: ${duration}s"
            return 1
        fi
    else
        log_error "Performance test failed: $response"
        return 1
    fi
}

# Main test function
main() {
    log_info "Starting Ollama integration tests for VCN Project..."
    echo ""
    
    # Run all tests
    run_test "Ollama Installation" test_ollama_installation
    run_test "Ollama Service" test_ollama_service
    run_test "Required Models" test_required_models
    run_test "Vision Model Functionality" test_vision_model
    run_test "Text Model Functionality" test_text_model
    run_test "Python Bridge" test_python_bridge
    run_test "Model Configuration" test_model_config
    run_test "API Endpoints" test_api_endpoints
    run_test "Concurrent Requests" test_concurrent_requests
    run_test "Performance Benchmark" test_performance
    
    # Display results
    echo ""
    log_info "Test Results:"
    log_success "Passed: $TESTS_PASSED"
    if [[ $TESTS_FAILED -gt 0 ]]; then
        log_error "Failed: $TESTS_FAILED"
    else
        log_info "Failed: $TESTS_FAILED"
    fi
    
    local total_tests=$((TESTS_PASSED + TESTS_FAILED))
    local success_rate=$((TESTS_PASSED * 100 / total_tests))
    
    echo ""
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All tests passed! (100%)"
        log_info "Ollama integration is working correctly"
        exit 0
    elif [[ $success_rate -ge 80 ]]; then
        log_warning "Most tests passed (${success_rate}%)"
        log_info "Ollama integration is mostly working"
        exit 0
    else
        log_error "Many tests failed (${success_rate}%)"
        log_error "Ollama integration needs attention"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    "quick")
        log_info "Running quick tests only..."
        run_test "Ollama Installation" test_ollama_installation
        run_test "Ollama Service" test_ollama_service
        run_test "Required Models" test_required_models
        ;;
    "performance")
        log_info "Running performance test only..."
        run_test "Performance Benchmark" test_performance
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [quick|performance|help]"
        echo ""
        echo "Commands:"
        echo "  quick       - Run only essential tests"
        echo "  performance - Run only performance benchmark"
        echo "  help        - Show this help message"
        echo ""
        echo "If no command is specified, all tests are run."
        ;;
    *)
        main "$@"
        ;;
esac 