#!/usr/bin/env python3
"""
Video Contextual Navigation (VCN) Production System Test
Validates that all components are working correctly
"""

import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path

def print_status(message, status="INFO"):
    """Print colored status message"""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌"
    }
    
    print(f"{colors.get(status, '')}{icons.get(status, '')} {message}{colors['RESET']}")

def run_command(command, timeout=30):
    """Run shell command and return result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)

def check_python_dependencies():
    """Check if all required Python packages are available"""
    print_status("Checking Python dependencies...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'requests': 'requests',
        'json': 'built-in',
        'base64': 'built-in',
        'time': 'built-in',
        'os': 'built-in'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print_status(f"  {package} ({pip_name})", "SUCCESS")
        except ImportError:
            print_status(f"  {package} ({pip_name}) - MISSING", "ERROR")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        print_status("Run: pip install -r requirements.txt", "INFO")
        return False
    
    return True

def check_ollama_installation():
    """Check if Ollama is installed and accessible"""
    print_status("Checking Ollama installation...")
    
    # Check if ollama command exists
    success, stdout, stderr = run_command("which ollama")
    if not success:
        print_status("Ollama not found in PATH", "ERROR")
        return False
    
    print_status(f"Ollama found: {stdout.strip()}", "SUCCESS")
    
    # Check ollama version
    success, stdout, stderr = run_command("ollama --version")
    if success:
        print_status(f"Ollama version: {stdout.strip()}", "SUCCESS")
    else:
        print_status("Could not get Ollama version", "WARNING")
    
    return True

def check_ollama_service():
    """Check if Ollama service is running"""
    print_status("Checking Ollama service...")
    
    try:
        response = requests.get("http://localhost:11434/", timeout=5)
        if response.status_code == 200:
            print_status("Ollama service is running", "SUCCESS")
            return True
        else:
            print_status(f"Ollama service returned status {response.status_code}", "WARNING")
            return False
    except requests.exceptions.ConnectionError:
        print_status("Ollama service is not running", "WARNING")
        return False
    except requests.exceptions.Timeout:
        print_status("Ollama service timeout", "WARNING")
        return False
    except Exception as e:
        print_status(f"Error checking Ollama service: {e}", "ERROR")
        return False

def start_ollama_service():
    """Start Ollama service if not running"""
    print_status("Starting Ollama service...")
    
    # Try to start ollama serve in background
    try:
        subprocess.Popen(
            ["ollama", "serve"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        
        # Wait for service to start
        for i in range(10):
            time.sleep(1)
            if check_ollama_service():
                print_status("Ollama service started successfully", "SUCCESS")
                return True
        
        print_status("Ollama service failed to start", "ERROR")
        return False
        
    except Exception as e:
        print_status(f"Error starting Ollama service: {e}", "ERROR")
        return False

def check_ollama_models():
    """Check if required models are available"""
    print_status("Checking Ollama models...")
    
    success, stdout, stderr = run_command("ollama list")
    if not success:
        print_status("Could not list Ollama models", "ERROR")
        return False
    
    models = stdout.lower()
    required_models = ["llava:7b"]
    optional_models = ["llama3.2-vision:11b"]
    
    available_models = []
    missing_models = []
    
    for model in required_models:
        if model in models:
            print_status(f"  {model} - Available", "SUCCESS")
            available_models.append(model)
        else:
            print_status(f"  {model} - Missing", "ERROR")
            missing_models.append(model)
    
    for model in optional_models:
        if model in models:
            print_status(f"  {model} - Available (optional)", "SUCCESS")
            available_models.append(model)
        else:
            print_status(f"  {model} - Missing (optional)", "WARNING")
    
    if missing_models:
        print_status(f"Missing required models: {', '.join(missing_models)}", "ERROR")
        print_status("Run: ollama pull llava:7b", "INFO")
        return False
    
    return True

def check_production_files():
    """Check if all production files are present"""
    print_status("Checking production files...")
    
    required_files = [
        "production_ready_integration.py",
        "requirements.txt",
        "README.md",
        "setup.sh"
    ]
    
    optional_files = [
        "test_video.mp4",
        "ollama_bridge.py",
        "TECHNICAL_ARCHITECTURE.md",
        "cpp_integration_example.cpp",
        "Makefile"
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print_status(f"  {file} - Present", "SUCCESS")
        else:
            print_status(f"  {file} - Missing", "ERROR")
            missing_files.append(file)
    
    for file in optional_files:
        if Path(file).exists():
            print_status(f"  {file} - Present (optional)", "SUCCESS")
        else:
            print_status(f"  {file} - Missing (optional)", "WARNING")
    
    if missing_files:
        print_status(f"Missing required files: {', '.join(missing_files)}", "ERROR")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with a simple frame analysis"""
    print_status("Testing basic functionality...")
    
    if not Path("test_video.mp4").exists():
        print_status("test_video.mp4 not found, skipping functionality test", "WARNING")
        return True
    
    print_status("Running quick test with 1 frame...")
    
    # Run the production script with minimal parameters
    command = "python3 production_ready_integration.py test_video.mp4 --frames 1 --json"
    success, stdout, stderr = run_command(command, timeout=300)  # 5 minute timeout
    
    if not success:
        print_status(f"Test failed: {stderr}", "ERROR")
        return False
    
    # Try to parse JSON output
    try:
        result = json.loads(stdout)
        if result.get("success", False):
            print_status("Basic functionality test passed", "SUCCESS")
            print_status(f"  Frames analyzed: {result.get('frames_analyzed', 'N/A')}", "INFO")
            print_status(f"  Processing time: {result.get('total_processing_time', 'N/A')}s", "INFO")
            print_status(f"  Model used: {result.get('model_used', 'N/A')}", "INFO")
            return True
        else:
            print_status(f"Test failed: {result.get('error', 'Unknown error')}", "ERROR")
            return False
    except json.JSONDecodeError:
        print_status(f"Invalid JSON output: {stdout[:100]}...", "ERROR")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🏭 Video Contextual Navigation (VCN) Production Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Python Dependencies", check_python_dependencies),
        ("Ollama Installation", check_ollama_installation),
        ("Production Files", check_production_files),
        ("Ollama Service", check_ollama_service),
        ("Ollama Models", check_ollama_models),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    # If Ollama service is not running, try to start it
    if not check_ollama_service():
        start_ollama_service()
    
    for test_name, test_func in tests:
        print_status(f"Running: {test_name}")
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print_status(f"Test '{test_name}' crashed: {e}", "ERROR")
            print()
    
    print("=" * 60)
    print_status(f"Test Results: {passed}/{total} tests passed", 
                 "SUCCESS" if passed == total else "WARNING")
    
    if passed == total:
        print_status("🎉 VCN Production System is ready!", "SUCCESS")
        print()
        print_status("Quick Start Commands:", "INFO")
        print("  python3 production_ready_integration.py test_video.mp4")
        print("  python3 production_ready_integration.py test_video.mp4 --json")
        print("  python3 production_ready_integration.py test_video.mp4 --frames 5")
    else:
        print_status("❌ Some tests failed. Please resolve issues before production use.", "ERROR")
        print()
        print_status("Common fixes:", "INFO")
        print("  pip install -r requirements.txt")
        print("  brew install ollama  # or curl -fsSL https://ollama.ai/install.sh | sh")
        print("  ollama pull llava:7b")
        print("  ./setup.sh  # Full automated setup")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 