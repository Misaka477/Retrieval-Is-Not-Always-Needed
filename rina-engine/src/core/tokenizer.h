#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Tokenizer interface: supports multiple tokenizer types (BBPE, SentencePiece...)
// Model loading: hybrid format (directory with weights.rinn + config + tokenizer/)
//   or single file (.rinn with embedded tokenizer)

enum class TokenizerType { None = 0, BBPE, SentencePiece };

struct Tokenizer {
    TokenizerType type = TokenizerType::None;
    std::unordered_map<int, std::string> id_to_token;   // for decode
    std::unordered_map<std::string, int> token_to_id;    // partial for encode (full is in merges)
    std::vector<std::pair<std::string, int>> vocab;      // (token, id) sorted by id for decode
    std::vector<std::string> special_tokens;
    
    // BBPE-specific
    std::vector<std::tuple<std::string, std::string, int>> merges; // (a, b, rank)
    
    // SentencePiece-specific
    float byte_fallback = false;
    
    bool load_dir(const std::string& dir_path);     // load tokenizer from directory
    bool load_json(const std::string& json_path);   // load from tokenizer.json
    
    std::string decode(int id) const;
    std::string decode(const std::vector<int>& ids) const;
    std::vector<int> encode(const std::string& text) const;
    
    int bos_id = -1;
    int vocab_size() const { return (int)id_to_token.size(); }
};

// Hybrid RINN model: load from directory or single file
struct RINNModel {
    std::string path;          // path to model.rinn/ (dir) or model.rinn (file)
    bool is_directory = false;
    Tokenizer tokenizer;
    
    bool load(const std::string& model_path);
    std::string weights_path() const;    // resolved path to weights.rinn
    std::string config_path() const;     // resolved path to config.json
};
