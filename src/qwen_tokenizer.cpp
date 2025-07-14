#include "qwen_tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <regex>

namespace qwen {

QwenTokenizer::QwenTokenizer(const std::string& tokenizer_path, const std::string& config_path) 
    : initialized_(false) {
    
    // Load SentencePiece model
    const auto status = processor_.Load(tokenizer_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load tokenizer model: " + tokenizer_path + 
                               " Error: " + status.ToString());
    }
    
    // Load configuration if provided
    if (!config_path.empty()) {
        load_config(config_path);
    }
    
    // Initialize special tokens
    initialize_special_tokens();
    
    initialized_ = true;
    std::cout << "Qwen tokenizer initialized successfully" << std::endl;
    std::cout << "Vocabulary size: " << config_.vocab_size << std::endl;
}

void QwenTokenizer::load_config(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cout << "Warning: Could not open config file: " << config_path 
                  << ". Using default configuration." << std::endl;
        return;
    }
    
    try {
        nlohmann::json config_json;
        file >> config_json;
        
        // Load special token IDs from config
        if (config_json.contains("bos_token_id")) {
            config_.bos_token_id = config_json["bos_token_id"];
        }
        if (config_json.contains("eos_token_id")) {
            config_.eos_token_id = config_json["eos_token_id"];
        }
        if (config_json.contains("image_token_id")) {
            config_.image_token_id = config_json["image_token_id"];
        }
        if (config_json.contains("video_token_id")) {
            config_.video_token_id = config_json["video_token_id"];
        }
        if (config_json.contains("vision_start_token_id")) {
            config_.vision_start_token_id = config_json["vision_start_token_id"];
        }
        if (config_json.contains("vision_end_token_id")) {
            config_.vision_end_token_id = config_json["vision_end_token_id"];
        }
        if (config_json.contains("vision_token_id")) {
            config_.vision_token_id = config_json["vision_token_id"];
        }
        if (config_json.contains("vocab_size")) {
            config_.vocab_size = config_json["vocab_size"];
        }
        
        std::cout << "Loaded configuration from: " << config_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Warning: Failed to parse config file: " << e.what() 
                  << ". Using default configuration." << std::endl;
    }
}

void QwenTokenizer::initialize_special_tokens() {
    // Map special token IDs to their string representations
    special_tokens_map_[config_.bos_token_id] = "<|endoftext|>";
    special_tokens_map_[config_.eos_token_id] = "<|im_end|>";
    special_tokens_map_[config_.image_token_id] = "<|image|>";
    special_tokens_map_[config_.video_token_id] = "<|video|>";
    special_tokens_map_[config_.vision_start_token_id] = "<|vision_start|>";
    special_tokens_map_[config_.vision_end_token_id] = "<|vision_end|>";
    special_tokens_map_[config_.vision_token_id] = "<|vision|>";
    special_tokens_map_[config_.pad_token_id] = "<|endoftext|>";
    
    // Create reverse mapping
    for (const auto& [id, token] : special_tokens_map_) {
        reverse_special_tokens_map_[token] = id;
    }
}

std::vector<int64_t> QwenTokenizer::encode(const std::string& text, bool add_special_tokens) {
    if (!initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    std::vector<int> token_ids;
    processor_.Encode(text, &token_ids);
    
    // Convert to int64_t
    std::vector<int64_t> result(token_ids.begin(), token_ids.end());
    
    if (add_special_tokens) {
        result = add_special_tokens_to_sequence(result);
    }
    
    return result;
}

std::string QwenTokenizer::decode(const std::vector<int64_t>& tokens, bool skip_special_tokens) {
    if (!initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    std::vector<int64_t> filtered_tokens = tokens;
    
    if (skip_special_tokens) {
        // Remove special tokens
        filtered_tokens.erase(
            std::remove_if(filtered_tokens.begin(), filtered_tokens.end(),
                          [this](int64_t token) { return is_special_token(token); }),
            filtered_tokens.end());
    }
    
    // Convert to int for SentencePiece
    std::vector<int> sp_tokens(filtered_tokens.begin(), filtered_tokens.end());
    
    std::string result;
    processor_.Decode(sp_tokens, &result);
    
    return result;
}

std::string QwenTokenizer::apply_chat_template(const nlohmann::json& messages, 
                                             bool add_generation_prompt,
                                             bool add_vision_id) {
    std::ostringstream template_str;
    
    // Add system prompt if not present
    bool has_system = false;
    for (const auto& message : messages) {
        if (message["role"] == "system") {
            has_system = true;
            break;
        }
    }
    
    if (!has_system) {
        template_str << config_.system_start 
                     << "You are a helpful assistant."
                     << config_.system_end;
    }
    
    int image_count = 0;
    int video_count = 0;
    
    // Process each message
    for (const auto& message : messages) {
        std::string role = message["role"];
        
        if (role == "system") {
            template_str << config_.system_start;
            if (message.contains("content") && message["content"].is_string()) {
                template_str << message["content"].get<std::string>();
            }
            template_str << config_.system_end;
        }
        else if (role == "user") {
            template_str << config_.user_start;
            
            if (message.contains("content")) {
                if (message["content"].is_string()) {
                    // Simple text content
                    template_str << message["content"].get<std::string>();
                }
                else if (message["content"].is_array()) {
                    // Multi-modal content
                    for (const auto& content_item : message["content"]) {
                        if (content_item["type"] == "text") {
                            template_str << content_item["text"].get<std::string>();
                        }
                        else if (content_item["type"] == "image") {
                            image_count++;
                            if (add_vision_id) {
                                template_str << "Picture " << image_count << ": ";
                            }
                            template_str << config_.vision_start 
                                        << config_.image_pad 
                                        << config_.vision_end;
                        }
                        else if (content_item["type"] == "video") {
                            video_count++;
                            if (add_vision_id) {
                                template_str << "Video " << video_count << ": ";
                            }
                            template_str << config_.vision_start 
                                        << config_.video_pad 
                                        << config_.vision_end;
                        }
                    }
                }
            }
            template_str << config_.user_end;
        }
        else if (role == "assistant") {
            template_str << config_.assistant_start;
            if (message.contains("content") && message["content"].is_string()) {
                template_str << message["content"].get<std::string>();
            }
            template_str << config_.assistant_end;
        }
    }
    
    // Add generation prompt if requested
    if (add_generation_prompt) {
        template_str << config_.assistant_start;
    }
    
    return template_str.str();
}

std::vector<int64_t> QwenTokenizer::create_vision_tokens(int num_image_tokens, int num_video_tokens) {
    std::vector<int64_t> tokens;
    
    // Add image tokens
    for (int i = 0; i < num_image_tokens; ++i) {
        tokens.push_back(config_.vision_start_token_id);
        tokens.push_back(config_.image_token_id);
        tokens.push_back(config_.vision_end_token_id);
    }
    
    // Add video tokens
    for (int i = 0; i < num_video_tokens; ++i) {
        tokens.push_back(config_.vision_start_token_id);
        tokens.push_back(config_.video_token_id);
        tokens.push_back(config_.vision_end_token_id);
    }
    
    return tokens;
}

std::string QwenTokenizer::format_vision_prompt(const std::string& text, 
                                               int num_images, 
                                               int num_videos) {
    std::ostringstream prompt;
    
    // Add vision tokens
    for (int i = 0; i < num_images; ++i) {
        prompt << config_.vision_start << config_.image_pad << config_.vision_end;
    }
    
    for (int i = 0; i < num_videos; ++i) {
        prompt << config_.vision_start << config_.video_pad << config_.vision_end;
    }
    
    // Add text
    prompt << text;
    
    return prompt.str();
}

QwenTokenizer::TokenizedInput QwenTokenizer::prepare_multimodal_input(
    const std::string& formatted_prompt, int max_length) {
    
    TokenizedInput result;
    
    // Tokenize the prompt
    result.input_ids = encode(formatted_prompt, true);
    
    // Truncate if necessary
    if (static_cast<int>(result.input_ids.size()) > max_length) {
        result.input_ids.resize(max_length);
    }
    
    result.sequence_length = static_cast<int64_t>(result.input_ids.size());
    
    // Create attention mask (1 for real tokens, 0 for padding)
    result.attention_mask.resize(result.sequence_length, 1);
    
    // Create position IDs
    result.position_ids.resize(result.sequence_length);
    for (int64_t i = 0; i < result.sequence_length; ++i) {
        result.position_ids[i] = i;
    }
    
    // Find image and video token positions
    for (int64_t i = 0; i < result.sequence_length; ++i) {
        if (result.input_ids[i] == config_.image_token_id) {
            result.image_positions.push_back(i);
        }
        else if (result.input_ids[i] == config_.video_token_id) {
            result.video_positions.push_back(i);
        }
    }
    
    return result;
}

bool QwenTokenizer::is_special_token(int64_t token_id) const {
    return special_tokens_map_.find(token_id) != special_tokens_map_.end();
}

std::string QwenTokenizer::token_to_string(int64_t token_id) const {
    auto it = special_tokens_map_.find(token_id);
    if (it != special_tokens_map_.end()) {
        return it->second;
    }
    
    // For regular tokens, decode through SentencePiece
    std::vector<int> sp_tokens = {static_cast<int>(token_id)};
    std::string result;
    processor_.Decode(sp_tokens, &result);
    return result;
}

int64_t QwenTokenizer::string_to_token(const std::string& token) const {
    auto it = reverse_special_tokens_map_.find(token);
    if (it != reverse_special_tokens_map_.end()) {
        return it->second;
    }
    
    // For regular tokens, encode through SentencePiece
    std::vector<int> sp_tokens;
    processor_.Encode(token, &sp_tokens);
    
    if (!sp_tokens.empty()) {
        return static_cast<int64_t>(sp_tokens[0]);
    }
    
    return -1; // Not found
}

std::string QwenTokenizer::create_chat_message(const std::string& role, const std::string& content) {
    if (role == "system") {
        return config_.system_start + content + config_.system_end;
    }
    else if (role == "user") {
        return config_.user_start + content + config_.user_end;
    }
    else if (role == "assistant") {
        return config_.assistant_start + content + config_.assistant_end;
    }
    
    return content; // Fallback
}

std::vector<int64_t> QwenTokenizer::add_special_tokens_to_sequence(const std::vector<int64_t>& tokens) {
    // For now, just return the tokens as-is
    // In a full implementation, you might want to add BOS/EOS tokens here
    return tokens;
}

// Utility functions implementation
namespace tokenizer_utils {

std::vector<float> create_attention_mask_tensor(const std::vector<int64_t>& attention_mask) {
    std::vector<float> result(attention_mask.size());
    std::transform(attention_mask.begin(), attention_mask.end(), result.begin(),
                  [](int64_t val) { return static_cast<float>(val); });
    return result;
}

std::vector<int64_t> create_position_ids_tensor(int64_t sequence_length, int64_t batch_size) {
    std::vector<int64_t> result(batch_size * sequence_length);
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < sequence_length; ++i) {
            result[b * sequence_length + i] = i;
        }
    }
    
    return result;
}

PositionInfo calculate_multimodal_positions(
    const QwenTokenizer::TokenizedInput& input,
    const std::vector<std::pair<int, int>>& image_grid_hw,
    const std::vector<std::tuple<int, int, int>>& video_grid_thw) {
    
    (void)image_grid_hw; // Mark as used to avoid warning
    (void)video_grid_thw; // Mark as used to avoid warning
    
    PositionInfo info;
    int64_t seq_len = input.sequence_length;
    
    // Initialize position arrays
    info.temporal_positions.resize(seq_len);
    info.height_positions.resize(seq_len);
    info.width_positions.resize(seq_len);
    
    // For text tokens, use linear positions
    for (int64_t i = 0; i < seq_len; ++i) {
        info.temporal_positions[i] = i;
        info.height_positions[i] = i;
        info.width_positions[i] = i;
    }
    
    // Calculate deltas based on vision content
    int64_t vision_tokens = static_cast<int64_t>(input.image_positions.size() + input.video_positions.size());
    int64_t max_position = seq_len - 1;
    
    // Simple delta calculation for now
    info.mrope_deltas.push_back(max_position + 1 - seq_len + vision_tokens);
    
    return info;
}

std::string format_describe_prompt(const std::string& question) {
    return "Describe this image in detail. " + question;
}

std::string format_analysis_prompt(const std::string& task_description) {
    return "Analyze the content of this image/video and " + task_description;
}

std::string format_extraction_prompt(const std::string& extraction_target) {
    return "Extract " + extraction_target + " from this image/video.";
}

} // namespace tokenizer_utils

} // namespace qwen 