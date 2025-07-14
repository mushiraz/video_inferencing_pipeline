#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

namespace qwen {

struct QwenTokenizerConfig {
    // Special token IDs based on config.json
    int64_t bos_token_id = 151643;
    int64_t eos_token_id = 151645;
    int64_t image_token_id = 151655;
    int64_t video_token_id = 151656;
    int64_t vision_start_token_id = 151652;
    int64_t vision_end_token_id = 151653;
    int64_t vision_token_id = 151654;
    int64_t pad_token_id = 151643; // Usually same as bos
    int64_t vocab_size = 151936;
    
    // Chat template tokens
    std::string system_start = "<|im_start|>system\n";
    std::string system_end = "<|im_end|>\n";
    std::string user_start = "<|im_start|>user\n";
    std::string user_end = "<|im_end|>\n";
    std::string assistant_start = "<|im_start|>assistant\n";
    std::string assistant_end = "<|im_end|>\n";
    
    // Vision placeholders
    std::string vision_start = "<|vision_start|>";
    std::string vision_end = "<|vision_end|>";
    std::string image_pad = "<|image_pad|>";
    std::string video_pad = "<|video_pad|>";
};

class QwenTokenizer {
public:
    explicit QwenTokenizer(const std::string& tokenizer_path, const std::string& config_path = "");
    ~QwenTokenizer() = default;

    // Core tokenization methods
    std::vector<int64_t> encode(const std::string& text, bool add_special_tokens = true);
    std::string decode(const std::vector<int64_t>& tokens, bool skip_special_tokens = true);
    
    // Chat template methods
    std::string apply_chat_template(const nlohmann::json& messages, 
                                  bool add_generation_prompt = false,
                                  bool add_vision_id = false);
    
    // Vision-specific tokenization
    std::vector<int64_t> create_vision_tokens(int num_image_tokens, int num_video_tokens = 0);
    std::string format_vision_prompt(const std::string& text, 
                                   int num_images = 0, 
                                   int num_videos = 0);
    
    // Multi-modal input preparation for ONNX
    struct TokenizedInput {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> position_ids;
        std::vector<int64_t> image_positions;
        std::vector<int64_t> video_positions;
        int64_t sequence_length;
    };
    
    TokenizedInput prepare_multimodal_input(const std::string& formatted_prompt,
                                           int max_length = 2048);
    
    // Utility methods
    bool is_special_token(int64_t token_id) const;
    std::string token_to_string(int64_t token_id) const;
    int64_t string_to_token(const std::string& token) const;
    
    // Getters
    const QwenTokenizerConfig& get_config() const { return config_; }
    int64_t get_vocab_size() const { return config_.vocab_size; }

private:
    void load_config(const std::string& config_path);
    void initialize_special_tokens();
    std::string create_chat_message(const std::string& role, const std::string& content);
    std::vector<int64_t> add_special_tokens_to_sequence(const std::vector<int64_t>& tokens);
    
    sentencepiece::SentencePieceProcessor processor_;
    QwenTokenizerConfig config_;
    std::unordered_map<int64_t, std::string> special_tokens_map_;
    std::unordered_map<std::string, int64_t> reverse_special_tokens_map_;
    bool initialized_;
};

// Helper functions for ONNX model preparation
namespace tokenizer_utils {
    
// Convert tokenized input to ONNX tensors
std::vector<float> create_attention_mask_tensor(const std::vector<int64_t>& attention_mask);
std::vector<int64_t> create_position_ids_tensor(int64_t sequence_length, int64_t batch_size = 1);

// Multi-modal position calculation for Qwen2-VL
struct PositionInfo {
    std::vector<int64_t> temporal_positions;  // For MRoPE
    std::vector<int64_t> height_positions;    // For MRoPE
    std::vector<int64_t> width_positions;     // For MRoPE
    std::vector<int64_t> mrope_deltas;        // Position deltas
};

PositionInfo calculate_multimodal_positions(const QwenTokenizer::TokenizedInput& input,
                                          const std::vector<std::pair<int, int>>& image_grid_hw,
                                          const std::vector<std::tuple<int, int, int>>& video_grid_thw);

// Format text for different prompt types
std::string format_describe_prompt(const std::string& question);
std::string format_analysis_prompt(const std::string& task_description);
std::string format_extraction_prompt(const std::string& extraction_target);

} // namespace tokenizer_utils

} // namespace qwen 