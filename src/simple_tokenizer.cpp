#include "simple_tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

namespace qwen {

SimpleTokenizer::SimpleTokenizer(const std::string& tokenizer_json_path, const std::string& config_path)
    : vocab_size_(151936), initialized_(false) {
    
    try {
        // Load tokenizer.json
        load_tokenizer_json(tokenizer_json_path);
        
        // Load config.json if provided
        if (!config_path.empty()) {
            load_config_json(config_path);
        }
        
        // Initialize special tokens
        special_tokens_[config_.bos_token_id] = "<|endoftext|>";
        special_tokens_[config_.eos_token_id] = "<|im_end|>";
        special_tokens_[config_.image_token_id] = "<|image|>";
        special_tokens_[config_.video_token_id] = "<|video|>";
        special_tokens_[config_.vision_start_token_id] = "<|vision_start|>";
        special_tokens_[config_.vision_end_token_id] = "<|vision_end|>";
        special_tokens_[config_.vision_token_id] = "<|vision|>";
        
        initialized_ = true;
        std::cout << "Simple tokenizer initialized successfully" << std::endl;
        std::cout << "Vocabulary size: " << vocab_size_ << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Warning: Failed to initialize tokenizer: " << e.what() << std::endl;
        std::cout << "Using minimal fallback tokenizer" << std::endl;
        initialized_ = false;
    }
}

void SimpleTokenizer::load_tokenizer_json(const std::string& tokenizer_json_path) {
    std::ifstream file(tokenizer_json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open tokenizer.json: " + tokenizer_json_path);
    }
    
    nlohmann::json tokenizer_json;
    file >> tokenizer_json;
    
    // Extract vocabulary from the model section
    if (tokenizer_json.contains("model") && tokenizer_json["model"].contains("vocab")) {
        auto vocab_obj = tokenizer_json["model"]["vocab"];
        
        for (auto it = vocab_obj.begin(); it != vocab_obj.end(); ++it) {
            std::string token = it.key();
            int64_t token_id = it.value();
            
            vocab_[token] = token_id;
            reverse_vocab_[token_id] = token;
        }
        
        std::cout << "Loaded " << vocab_.size() << " vocabulary entries" << std::endl;
    } else {
        std::cout << "Warning: Could not find vocab in tokenizer.json" << std::endl;
    }
}

void SimpleTokenizer::load_config_json(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cout << "Warning: Could not open config.json: " << config_path << std::endl;
        return;
    }
    
    try {
        nlohmann::json config_json;
        file >> config_json;
        
        // Load special token IDs
        if (config_json.contains("bos_token_id")) {
            config_.bos_token_id = config_json["bos_token_id"];
        }
        if (config_json.contains("eos_token_id")) {
            config_.eos_token_id = config_json["eos_token_id"];
        }
        if (config_json.contains("vocab_size")) {
            config_.vocab_size = config_json["vocab_size"];
            vocab_size_ = config_.vocab_size;
        }
        
        std::cout << "Loaded configuration from: " << config_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Warning: Failed to parse config.json: " << e.what() << std::endl;
    }
}

std::vector<int64_t> SimpleTokenizer::encode(const std::string& text, bool add_special_tokens) {
    if (!initialized_) {
        return simple_encode_fallback(text);
    }
    
    std::vector<int64_t> tokens;
    
    // Simple whitespace tokenization
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // Look for exact matches first
        auto it = vocab_.find(word);
        if (it != vocab_.end()) {
            tokens.push_back(it->second);
        } else {
            // Look for subwords or characters
            for (char c : word) {
                std::string char_str(1, c);
                auto char_it = vocab_.find(char_str);
                if (char_it != vocab_.end()) {
                    tokens.push_back(char_it->second);
                } else {
                    // Use unknown token or approximate
                    tokens.push_back(100); // Approximate unknown token ID
                }
            }
        }
    }
    
    if (add_special_tokens && !tokens.empty()) {
        // Add BOS token at the beginning
        tokens.insert(tokens.begin(), config_.bos_token_id);
    }
    
    return tokens;
}

std::string SimpleTokenizer::decode(const std::vector<int64_t>& tokens, bool skip_special_tokens) {
    std::string result;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        int64_t token_id = tokens[i];
        
        if (skip_special_tokens && is_special_token(token_id)) {
            continue;
        }
        
        // Look up token
        auto it = reverse_vocab_.find(token_id);
        if (it != reverse_vocab_.end()) {
            std::string token_text = it->second;
            
            // Handle BPE space marker (Ġ represents a space in GPT tokenizers)
            if (token_text.length() >= 2 && token_text.substr(0, 2) == "Ġ") {
                // Replace Ġ with actual space and append
                if (!result.empty()) {
                    result += " ";
                }
                result += token_text.substr(2); // Remove the Ġ character (2 bytes in UTF-8)
            } else {
                // Regular token - just append (no space for BPE continuation)
                result += token_text;
            }
        } else {
            // Check special tokens
            auto special_it = special_tokens_.find(token_id);
            if (special_it != special_tokens_.end()) {
                result += special_it->second;
            } else {
                // Unknown token - try to provide meaningful fallback
                if (token_id >= 0 && token_id < 50000) {
                    // Likely a valid token ID, just unknown to us
                    result += "[" + std::to_string(token_id) + "]";
                } else {
                    result += "<unk>";
                }
            }
        }
    }
    
    // Clean up any BPE artifacts and normalize whitespace
    std::string cleaned_result;
    bool last_was_space = false;
    
    for (char c : result) {
        if (c == ' ') {
            if (!last_was_space && !cleaned_result.empty()) {
                cleaned_result += ' ';
                last_was_space = true;
            }
        } else {
            cleaned_result += c;
            last_was_space = false;
        }
    }
    
    return cleaned_result;
}

bool SimpleTokenizer::is_special_token(int64_t token_id) const {
    return special_tokens_.find(token_id) != special_tokens_.end();
}

std::vector<int64_t> SimpleTokenizer::simple_encode_fallback(const std::string& text) {
    // Very simple fallback - just map characters to ASCII values + offset
    std::vector<int64_t> tokens;
    
    for (char c : text) {
        if (c == ' ') {
            tokens.push_back(220); // Common space token
        } else if (c >= 32 && c <= 126) {
            tokens.push_back(static_cast<int64_t>(c) + 1000); // Offset ASCII
        }
    }
    
    return tokens;
}

} // namespace qwen 