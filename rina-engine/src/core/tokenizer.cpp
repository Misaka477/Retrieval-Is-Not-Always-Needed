#include "tokenizer.h"
#include "json.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if(!f) return {};
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

// ── load from tokenizer.json (HuggingFace BBPE format) ──

static bool load_bbpe(Tokenizer* t, const std::string& json_path) {
    auto raw = read_file(json_path);
    if(raw.empty()) return false;
    json tok;
    try { tok = json::parse(raw); } catch(...){ return false; }
    
    auto& model = tok["model"];
    if(model.is_null()) return false;
    
    // Parse vocab
    auto& vocab = model["vocab"];
    if(vocab.is_null()) return false;
    
    t->id_to_token.clear(); t->token_to_id.clear();
    for(auto& [token_str, id_val] : vocab.items()){
        int id = id_val.get<int>();
        t->id_to_token[id] = token_str;
        t->token_to_id[token_str] = id;
    }
    
    // Parse merges
    t->merges.clear();
    auto& merges = model["merges"];
    if(!merges.is_null()){
        for(size_t i=0; i<merges.size(); i++){
            std::string m = merges[i].get<std::string>();
            auto sp = m.find(' ');
            if(sp != std::string::npos)
                t->merges.emplace_back(m.substr(0,sp), m.substr(sp+1), (int)i);
        }
    }
    
    // Parse added tokens (BOS, EOS, special tokens)
    auto& added = tok["added_tokens"];
    if(!added.is_null()){
        for(auto& at : added){
            int id = at["id"].get<int>();
            std::string content = at["content"].get<std::string>();
            t->id_to_token[id] = content;
            t->token_to_id[content] = id;
            // Check for BOS token (first added token or explicitly marked)
            auto s  = at.find("special");
            if(s != at.end() && s->get<bool>() && t->bos_id < 0)
                t->bos_id = id;
        }
    }
    // Fallback: read bos_token from root config
    if(t->bos_id < 0 && tok["bos_token"].is_string()){
        std::string bt = tok["bos_token"].get<std::string>();
        auto it = t->token_to_id.find(bt);
        if(it != t->token_to_id.end()) t->bos_id = it->second;
    }

    t->type = TokenizerType::BBPE;
    fprintf(stderr,"tokenizer: BBPE loaded (vocab=%zu, merges=%zu, added=%zu)\n",
            t->id_to_token.size(), t->merges.size(),
            added.is_null()?0:added.size());
    return true;
}

// ── Tokenizer: load ──

bool Tokenizer::load_dir(const std::string& dir) {
    // Try tokenizer.json
    auto tj = dir + "/tokenizer.json";
    if(fs::exists(tj) && load_bbpe(this, tj)) return true;
    
    fprintf(stderr,"tokenizer: no supported tokenizer found in %s\n", dir.c_str());
    return false;
}

bool Tokenizer::load_json(const std::string& path) {
    return load_bbpe(this, path);
}

// ── decode ──

static std::string unescape(const std::string& s) {
    std::string r = s;
    // BPE saves spaces as Ġ (U+0120). Replace with ASCII space.
    for (size_t i = 0; i < r.size(); i++)
        if ((unsigned char)r[i] == 0xC4 && i+1 < r.size() && (unsigned char)r[i+1] == 0xA0) {
            r[i] = ' '; r.erase(i+1, 1);
        }
    return r;
}

std::string Tokenizer::decode(int id) const {
    auto it = id_to_token.find(id);
    if(it==id_to_token.end()) return "";
    return unescape(it->second);
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string r;
    for(int id : ids){
        auto it = id_to_token.find(id);
        if(it!=id_to_token.end()) r += unescape(it->second);
    }
    return r;
}

// ── encode (standard BPE) ──

static int merge_rank(const std::vector<std::tuple<std::string,std::string,int>>& merges,
                       const std::string& a, const std::string& b) {
    for(auto& m : merges)
        if(std::get<0>(m)==a && std::get<1>(m)==b) return std::get<2>(m);
    return -1;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if(id_to_token.empty() || merges.empty()) return {};
    
    // Start with byte-level tokens
    std::vector<std::string> tokens;
    for(unsigned char c : text){
        std::string s(1, (char)c);
        if(token_to_id.count(s)) tokens.push_back(s);
    }
    
    // Greedy BPE merge
    while(true){
        int best_rank = 1000000000; int best_i = -1;
        for(int i=0; i+1<(int)tokens.size(); i++){
            int r = merge_rank(merges, tokens[i], tokens[i+1]);
            if(r>=0 && r<best_rank){ best_rank=r; best_i=i; }
        }
        if(best_i<0) break;
        tokens[best_i] = tokens[best_i] + tokens[best_i+1];
        tokens.erase(tokens.begin() + best_i + 1);
    }
    
    std::vector<int> ids;
    if(bos_id >= 0) ids.push_back(bos_id);
    for(auto& t : tokens){
        auto it = token_to_id.find(t);
        if(it!=token_to_id.end()) ids.push_back(it->second);
    }
    return ids;
}

// ── RINN model loading ──

bool RINNModel::load(const std::string& model_path) {
    path = model_path;
    is_directory = fs::is_directory(model_path);
    
    if(is_directory){
        // Load tokenizer from model directory
        std::string tok_dir = model_path + "/tokenizer";
        if(fs::exists(tok_dir)) tokenizer.load_dir(tok_dir);
        else if(fs::exists(model_path + "/tokenizer.json"))
            tokenizer.load_dir(model_path);  // load from model root dir
    }
    return true;
}

std::string RINNModel::weights_path() const {
    if(is_directory) return path + "/weights.rinn";
    return path;
}

std::string RINNModel::config_path() const {
    if(is_directory) return path + "/config.json";
    return "";
}
