#!/bin/bash
# GPU-Accelerated Video Analysis Setup Script
# For AMD GPUs with ROCm support

set -e

echo "🎮 Setting up GPU-Accelerated Video Analysis Pipeline"
echo "======================================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root for safety"
   exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VERSION=$VERSION_ID
else
    echo "❌ Cannot detect OS version"
    exit 1
fi

echo "🖥️  Detected OS: $OS $VERSION"

# Function to install ROCm on Ubuntu/Debian
install_rocm_ubuntu() {
    echo "📦 Installing ROCm for Ubuntu/Debian..."
    
    # Add ROCm repository
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
    
    sudo apt update
    
    # Install ROCm packages
    sudo apt install -y \
        rocm-dev \
        rocm-utils \
        rocm-smi \
        hip-dev \
        hipcc \
        rocdecode-dev \
        rocjpeg-dev
    
    # Add user to render group
    sudo usermod -a -G render $USER
    sudo usermod -a -G video $USER
    
    echo "✅ ROCm installed successfully"
    echo "⚠️  Please reboot to activate group memberships"
}

# Function to install ROCm on RHEL/CentOS
install_rocm_rhel() {
    echo "📦 Installing ROCm for RHEL/CentOS..."
    
    # Add ROCm repository
    sudo tee /etc/yum.repos.d/rocm.repo <<EOF
[rocm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/yum/rpm
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
    
    # Install ROCm packages
    sudo dnf install -y \
        rocm-dev \
        rocm-smi \
        hip-devel \
        hipcc \
        rocdecode-devel \
        rocjpeg-devel
    
    # Add user to render group
    sudo usermod -a -G render $USER
    sudo usermod -a -G video $USER
    
    echo "✅ ROCm installed successfully"
    echo "⚠️  Please reboot to activate group memberships"
}

# Check for GPU hardware
echo "🔍 Checking AMD GPU hardware..."
if lspci | grep -i amd | grep -i vga > /dev/null; then
    echo "✅ AMD GPU detected"
    lspci | grep -i amd | grep -i vga
else
    echo "⚠️  No AMD GPU detected - proceeding with CPU fallback support"
fi

# Install ROCm based on OS
case "$OS" in
    *"Ubuntu"*|*"Debian"*)
        install_rocm_ubuntu
        ;;
    *"Red Hat"*|*"CentOS"*|*"Fedora"*)
        install_rocm_rhel
        ;;
    *)
        echo "⚠️  Unsupported OS for automatic ROCm installation"
        echo "Please install ROCm manually: https://rocmdocs.amd.com/en/latest/Installation_Guide/"
        ;;
esac

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if command -v python3 &> /dev/null; then
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements_gpu.txt
    echo "✅ Python dependencies installed"
else
    echo "❌ Python 3 not found - please install Python 3.8+"
    exit 1
fi

# Check FFmpeg installation (for CPU fallback)
echo "🎬 Checking FFmpeg installation..."
if command -v ffmpeg &> /dev/null && command -v ffprobe &> /dev/null; then
    echo "✅ FFmpeg found: $(ffmpeg -version | head -n1)"
else
    echo "📦 Installing FFmpeg..."
    case "$OS" in
        *"Ubuntu"*|*"Debian"*)
            sudo apt update && sudo apt install -y ffmpeg
            ;;
        *"Red Hat"*|*"CentOS"*|*"Fedora"*)
            sudo dnf install -y ffmpeg
            ;;
    esac
fi

# Verify GPU setup
echo "🎮 Verifying GPU setup..."
if command -v rocm-smi &> /dev/null; then
    echo "GPU Information:"
    rocm-smi || echo "⚠️  rocm-smi failed - may need reboot"
else
    echo "⚠️  rocm-smi not found - ROCm may not be properly installed"
fi

# Test Python imports
echo "🧪 Testing Python imports..."
python3 -c "
import numpy as np
import requests
from PIL import Image
print('✅ Core dependencies working')

try:
    import cv2
    print('✅ OpenCV available for CPU fallback')
except ImportError:
    print('⚠️  OpenCV not available - using PIL only')

try:
    import scipy
    print('✅ SciPy available for image processing')
except ImportError:
    print('⚠️  SciPy not available - using basic resizing')
"

# Set up environment
echo "⚙️  Setting up environment..."
cat > gpu_env_setup.sh << 'EOF'
#!/bin/bash
# GPU Environment Setup
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100

echo "🎮 GPU Environment configured"
rocm-smi || echo "⚠️  GPU status unavailable"
EOF

chmod +x gpu_env_setup.sh

# Create test script
echo "🧪 Creating GPU test script..."
cat > test_gpu_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test GPU setup for video analysis pipeline"""

import sys
import ctypes
import subprocess

def test_rocm_installation():
    """Test ROCm installation and GPU availability"""
    print("🎮 Testing ROCm Installation")
    print("=" * 40)
    
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ rocm-smi working")
            print(result.stdout)
        else:
            print("❌ rocm-smi failed")
            return False
    except FileNotFoundError:
        print("❌ rocm-smi not found")
        return False
    
    return True

def test_library_loading():
    """Test rocDecode and rocJPEG library loading"""
    print("\n🔧 Testing Library Loading")
    print("=" * 40)
    
    # Test rocDecode
    rocdecode_paths = [
        "/opt/rocm/lib/librocdecode.so",
        "/usr/lib/x86_64-linux-gnu/librocdecode.so",
        "librocdecode.so"
    ]
    
    rocdecode_found = False
    for path in rocdecode_paths:
        try:
            lib = ctypes.CDLL(path)
            print(f"✅ rocDecode loaded from: {path}")
            rocdecode_found = True
            break
        except OSError:
            continue
    
    if not rocdecode_found:
        print("❌ rocDecode library not found")
    
    # Test rocJPEG
    rocjpeg_paths = [
        "/opt/rocm/lib/librocjpeg.so",
        "/usr/lib/x86_64-linux-gnu/librocjpeg.so",
        "librocjpeg.so"
    ]
    
    rocjpeg_found = False
    for path in rocjpeg_paths:
        try:
            lib = ctypes.CDLL(path)
            print(f"✅ rocJPEG loaded from: {path}")
            rocjpeg_found = True
            break
        except OSError:
            continue
    
    if not rocjpeg_found:
        print("❌ rocJPEG library not found")
    
    return rocdecode_found and rocjpeg_found

def test_python_dependencies():
    """Test Python dependencies"""
    print("\n🐍 Testing Python Dependencies")
    print("=" * 40)
    
    required_modules = ['numpy', 'requests', 'PIL', 'ctypes']
    optional_modules = ['cv2', 'scipy']
    
    all_good = True
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - REQUIRED")
            all_good = False
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"⚠️  {module} - optional")
    
    return all_good

def main():
    print("🎮 GPU-Accelerated Video Analysis Setup Test")
    print("=" * 50)
    
    rocm_ok = test_rocm_installation()
    libs_ok = test_library_loading()
    python_ok = test_python_dependencies()
    
    print("\n📊 Summary")
    print("=" * 20)
    
    if rocm_ok and libs_ok and python_ok:
        print("🎉 GPU acceleration ready!")
        print("✅ All components working")
        return 0
    elif python_ok:
        print("💻 CPU fallback ready")
        print("⚠️  GPU libraries not available - will use CPU mode")
        return 0
    else:
        print("❌ Setup incomplete")
        print("Please check missing dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_gpu_setup.py

echo ""
echo "🎉 GPU-Accelerated Setup Complete!"
echo "=" * 40
echo "Next steps:"
echo "1. Reboot system to activate group memberships"
echo "2. Run: source gpu_env_setup.sh"
echo "3. Test: python3 test_gpu_setup.py"
echo "4. Run: python3 gpu_accelerated_integration.py your_video.mp4"
echo ""
echo "💡 GPU Mode: Uses rocDecode + rocJPEG for maximum performance"
echo "💻 CPU Fallback: Automatically used if GPU libraries unavailable" 