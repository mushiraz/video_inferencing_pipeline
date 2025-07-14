# Video Contextual Navigation (VCN) Production Makefile
# Builds C++ integration example for the VCN system

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Operating system detection
UNAME_S := $(shell uname -s)

# Include paths for different systems
ifeq ($(UNAME_S),Darwin)
    # macOS with Homebrew
    INCLUDES = -I/opt/homebrew/include -I/usr/local/include
    LDFLAGS = 
    LIBS = 
    PLATFORM = macOS
else ifeq ($(UNAME_S),Linux)
    # Linux (Ubuntu/Debian)
    INCLUDES = -I/usr/include/nlohmann
    LDFLAGS = 
    LIBS = 
    PLATFORM = Linux
else
    # Default
    INCLUDES = 
    LDFLAGS = 
    LIBS = 
    PLATFORM = Unknown
endif

# Target binary
TARGET = vcn_analyzer

# Source files
SOURCES = cpp_integration_example.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: info $(TARGET)

# System information
info:
	@echo "=============================================="
	@echo "Building VCN C++ Integration for $(PLATFORM)"
	@echo "=============================================="
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "Includes: $(INCLUDES)"
	@echo ""

# Build the main target
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "✅ Built $(TARGET) successfully!"
	@echo ""
	@echo "Usage:"
	@echo "  ./$(TARGET) test_video.mp4"
	@echo "  ./$(TARGET) video1.mp4 video2.mp4"
	@echo ""

# Compile source files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Install dependencies
deps:
	@echo "Installing dependencies for $(PLATFORM)..."
ifeq ($(UNAME_S),Darwin)
	@echo "Installing via Homebrew..."
	brew install nlohmann-json || echo "nlohmann-json may already be installed"
else ifeq ($(UNAME_S),Linux)
	@echo "Installing via apt (requires sudo)..."
	sudo apt-get update
	sudo apt-get install nlohmann-json3-dev
else
	@echo "Please install nlohmann-json manually for your system"
endif
	@echo "✅ Dependencies installed!"

# Test the system
test: $(TARGET)
	@echo "Testing VCN C++ Integration..."
	@if [ -f test_video.mp4 ]; then \
		echo "Running test with sample video..."; \
		./$(TARGET) test_video.mp4; \
	else \
		echo "⚠️  test_video.mp4 not found, skipping test"; \
		echo "Please provide a test video file"; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(TARGET)
	@echo "✅ Clean complete!"

# Install the binary (optional)
install: $(TARGET)
	@echo "Installing $(TARGET) to /usr/local/bin..."
	sudo cp $(TARGET) /usr/local/bin/
	@echo "✅ $(TARGET) installed system-wide!"
	@echo "You can now run: vcn_analyzer video.mp4"

# Uninstall the binary
uninstall:
	@echo "Uninstalling $(TARGET)..."
	sudo rm -f /usr/local/bin/$(TARGET)
	@echo "✅ $(TARGET) uninstalled!"

# Setup entire VCN system
setup: deps
	@echo "Setting up complete VCN system..."
	@if [ -f setup.sh ]; then \
		chmod +x setup.sh; \
		./setup.sh; \
	else \
		echo "⚠️  setup.sh not found"; \
	fi

# Check system requirements
check:
	@echo "Checking VCN system requirements..."
	@echo ""
	@echo "🔍 Python 3:"
	@which python3 || echo "❌ Python 3 not found"
	@python3 --version || echo "❌ Python 3 not working"
	@echo ""
	@echo "🔍 Ollama:"
	@which ollama || echo "❌ Ollama not found"
	@echo ""
	@echo "🔍 OpenCV (Python):"
	@python3 -c "import cv2; print('✅ OpenCV version:', cv2.__version__)" || echo "❌ OpenCV not found"
	@echo ""
	@echo "🔍 Requests (Python):"
	@python3 -c "import requests; print('✅ Requests available')" || echo "❌ Requests not found"
	@echo ""
	@echo "🔍 C++ Compiler:"
	@$(CXX) --version | head -1 || echo "❌ C++ compiler not found"
	@echo ""
	@echo "🔍 nlohmann/json:"
	@echo "#include <nlohmann/json.hpp>" | $(CXX) $(INCLUDES) -x c++ -c - -o /dev/null 2>/dev/null && echo "✅ nlohmann/json available" || echo "❌ nlohmann/json not found"

# Help target
help:
	@echo "VCN Production System Makefile"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  all      - Build the C++ integration binary"
	@echo "  deps     - Install C++ dependencies"
	@echo "  test     - Build and test the system"
	@echo "  setup    - Complete system setup (deps + Python setup)"
	@echo "  check    - Check system requirements"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install binary system-wide"
	@echo "  uninstall- Remove installed binary"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Example workflow:"
	@echo "  make deps     # Install dependencies"
	@echo "  make setup    # Full system setup"
	@echo "  make all      # Build C++ integration"
	@echo "  make test     # Test the system"
	@echo ""

# Phony targets
.PHONY: all info deps test clean install uninstall setup check help 