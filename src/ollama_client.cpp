// Stub implementation for ollama_client.cpp
// This file is created to satisfy CMake build requirements

#include <string>
#include <iostream>

class OllamaClient {
public:
    OllamaClient() = default;
    ~OllamaClient() = default;
    
    bool isAvailable() const {
        return false;
    }
    
    std::string query(const std::string& prompt) const {
        return "Ollama not implemented";
    }
};

// Stub functions to satisfy linker
extern "C" {
    void ollama_init() {
        // Stub implementation
    }
    
    void ollama_cleanup() {
        // Stub implementation
    }
} 