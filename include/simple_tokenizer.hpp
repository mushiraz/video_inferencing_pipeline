#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <regex>

namespace qwen {

class SimpleTokenizer {
public:
    explicit SimpleTokenizer(const std::string& tokenizer_json_path, const std::string& config_path = "");
    ~SimpleTokenizer() = default;

    // Core tokenization methods
    std::vector<int64_t> encode(const std::string& text, bool add_special_tokens = true);
    std::string decode(const std::vector<int64_t>& tokens, bool skip_special_tokens = true);
    
    // Special token handling
    bool is_special_token(int64_t token_id) const;
    int64_t get_vocab_size() const { return vocab_size_; }
    
    // Config access
    struct Config {
        int64_t bos_token_id = 151643;
        int64_t eos_token_id = 151645;
        int64_t image_token_id = 151655;
        int64_t video_token_id = 151656;
        int64_t vision_start_token_id = 151652;
        int64_t vision_end_token_id = 151653;
        int64_t vision_token_id = 151654;
        int64_t pad_token_id = 151643;
        int64_t vocab_size = 151936;
    };
    
    const Config& get_config() const { return config_; }

private:
    void load_tokenizer_json(const std::string& tokenizer_json_path);
    void load_config_json(const std::string& config_path);
    std::vector<int64_t> simple_encode_fallback(const std::string& text);
    
    std::unordered_map<std::string, int64_t> vocab_;
    std::unordered_map<int64_t, std::string> reverse_vocab_;
    std::unordered_map<int64_t, std::string> special_tokens_;
    Config config_;
    int64_t vocab_size_;
    bool initialized_;
};

} // namespace qwen 