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

// ── GPT-2 ByteLevel BPE bytes_to_unicode table ──
// Builds the mapping: byte (0-255) → unicode character string
// and reverse: unicode codepoint → byte
// Matches GPT-2 / llama.cpp / transformers implementation
static void build_byte_unicode_tables(Tokenizer* t) {
    // bytes that map to themselves (printable ASCII + Latin-1 supplement)
    std::vector<int> bs;
    for (int i = 33; i <= 126; i++) bs.push_back(i);   // printable ASCII
    for (int i = 161; i <= 172; i++) bs.push_back(i);  // Latin-1 supplement
    for (int i = 174; i <= 255; i++) bs.push_back(i);  // Latin-1 supplement (cont)
    
    std::fill(t->unicode_to_byte, t->unicode_to_byte + 512, 0xFF);
    
    int n = 0;
    for (int b = 0; b < 256; b++) {
        auto it = std::find(bs.begin(), bs.end(), b);
        if (it != bs.end()) {
            // byte maps to itself
            t->byte_to_unicode[b] = std::string(1, (char)b);
            t->unicode_to_byte[b] = (uint8_t)b;
        } else {
            // byte maps to unicode at 256 + n
            int codepoint = 256 + n;
            n++;
            // Encode codepoint to UTF-8
            if (codepoint < 0x80) {
                t->byte_to_unicode[b] = std::string(1, (char)codepoint);
            } else if (codepoint < 0x800) {
                t->byte_to_unicode[b] = {
                    (char)(0xC0 | (codepoint >> 6)),
                    (char)(0x80 | (codepoint & 0x3F))
                };
            } else {
                t->byte_to_unicode[b] = {
                    (char)(0xE0 | (codepoint >> 12)),
                    (char)(0x80 | ((codepoint >> 6) & 0x3F)),
                    (char)(0x80 | (codepoint & 0x3F))
                };
            }
            t->unicode_to_byte[codepoint] = (uint8_t)b;
        }
    }
}

// Decode a single UTF-8 character from string at given position.
// Returns the codepoint and advances pos past the encoded bytes.
// Returns -1 on error.
static int decode_utf8(const std::string& s, size_t& pos) {
    if (pos >= s.size()) return -1;
    unsigned char c0 = (unsigned char)s[pos];
    if (c0 < 0x80) { pos++; return c0; }
    if (c0 < 0xC0) { pos++; return -1; } // unexpected continuation byte
    if (pos + 1 >= s.size()) { pos++; return -1; }
    unsigned char c1 = (unsigned char)s[pos + 1];
    if (c0 < 0xE0) { pos += 2; return ((c0 & 0x1F) << 6) | (c1 & 0x3F); }
    if (pos + 2 >= s.size()) { pos += 2; return -1; }
    unsigned char c2 = (unsigned char)s[pos + 2];
    pos += 3;
    return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
}

// ── Decode: convert token string back to text ──
// GPT-2 BPE encodes bytes as unicode chars via bytes_to_unicode table.
// Decode reverses this: map each unicode char back to its original byte.
static std::string bbpe_decode(const Tokenizer* t, const std::string& token) {
    std::string r;
    size_t pos = 0;
    while (pos < token.size()) {
        size_t start = pos;
        int cp = decode_utf8(token, pos);
        if (cp < 0) {
            r += token[start]; // bad UTF-8, pass through
            continue;
        }
        if (cp < 512 && t->unicode_to_byte[cp] != 0xFF) {
            r += (char)t->unicode_to_byte[cp];
        } else {
            // Not a mapped byte, append original UTF-8 bytes
            r.append(token.data() + start, pos - start);
        }
    }
    return r;
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

    // Build bytes_to_unicode tables for GPT-2 BPE decode
    build_byte_unicode_tables(t);

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

std::string Tokenizer::decode(int id) const {
    auto it = id_to_token.find(id);
    if(it==id_to_token.end()) return "";
    return bbpe_decode(this, it->second);
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string r;
    for(int id : ids){
        auto it = id_to_token.find(id);
        if(it!=id_to_token.end()) r += bbpe_decode(this, it->second);
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
    
    // Start with byte-level tokens via bytes_to_unicode mapping
    std::vector<std::string> tokens;
    for(unsigned char c : text){
        const std::string& mapped = byte_to_unicode[c];
        if(token_to_id.count(mapped)) {
            tokens.push_back(mapped);
        } else if (!mapped.empty()) {
            tokens.push_back(mapped); // still use it, will fail lookup later
        }
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
    
    auto try_load = [this](const std::string& dir) {
        if(fs::exists(dir + "/tokenizer.json")) tokenizer.load_dir(dir);
    };
    
    if(is_directory){
        try_load(model_path);
    } else {
        // .rinn file: try tokenizer.json next to it
        auto parent = fs::path(model_path).parent_path().string();
        if(!parent.empty()) try_load(parent);
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
