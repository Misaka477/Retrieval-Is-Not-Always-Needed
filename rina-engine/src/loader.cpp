#include "core/config.h"
#include "core/tensor.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <sys/stat.h>

struct RinnTensorSpec { std::string name; int shape[4]; int n_dim; int quant_type; size_t size; size_t offset; };

static int read_int32(const uint8_t* p) { return p[0] | (p[1]<<8) | (p[2]<<16) | (p[3]<<24); }
static bool is_directory(const char* path) {
    struct stat st; stat(path, &st); return S_ISDIR(st.st_mode);
}
static std::string read_config(const char* path_in, ModelConfig& cfg, TensorMap& tensors) {
    std::string path_str = path_in;
    if(is_directory(path_in)) path_str += "/weights.rinn";
    const char* path = path_str.c_str();
    FILE* f = fopen(path, "rb"); if (!f) return "";
    char hdr[512]; fread(hdr, 1, 512, f);
    if (memcmp(hdr, "RINN", 4) != 0) { fclose(f); return ""; }
    int version; memcpy(&version, hdr+4, 4);
    int n_tensors; memcpy(&n_tensors, hdr+8, 4);
    size_t cfg_off, cfg_sz, idx_off, idx_sz, data_off;
    memcpy(&cfg_off, hdr+16, 8); memcpy(&cfg_sz, hdr+24, 8);
    memcpy(&idx_off, hdr+32, 8); memcpy(&idx_sz, hdr+40, 8);
    memcpy(&data_off, hdr+48, 8);

    std::string json(cfg_sz, '\0');
    fseek(f, cfg_off, SEEK_SET); fread(&json[0], 1, cfg_sz, f);

    auto gk = [&](const std::string& k) -> std::string {
        auto p = json.find("\""+k+"\""); if (p==std::string::npos) return "";
        p = json.find(':', p+k.size()+2); if (p==std::string::npos) return ""; p++;
        while (p<json.size() && (json[p]==' '||json[p]=='\t'||json[p]=='\n'||json[p]=='\r')) p++;
        if (json[p]=='"') { auto e=json.find('"',p+1); return json.substr(p+1,e-p-1); }
        auto e=p; while(e<json.size()&&json[e]!=','&&json[e]!='}'&&json[e]!='\n') e++;
        return json.substr(p,e-p);
    };
    auto gi = [&](const std::string& k) -> int {
        auto v=gk(k); return v.empty()?0:std::stoi(v);
    };

    cfg.name = gk("name");
    cfg.dim = gi("dim"); cfg.n_layers = gi("n_layers"); cfg.n_heads = gi("n_heads");
    cfg.n_kv_heads = gi("n_kv_heads"); cfg.head_dim = gi("head_dim");
    cfg.d_c = gi("d_c"); cfg.d_h_r = gi("d_h_r");
    cfg.vocab_size = gi("vocab_size"); cfg.max_seq_len = gi("block_size");
    cfg.ssm_steps = gi("ssm_steps");

    // Parse layers
    std::string layers_str;
    auto lp = json.find("\"layers\""); if (lp!=std::string::npos) {
        auto as = json.find('[',lp); auto ae = json.find(']',as);
        if (as!=std::string::npos && ae!=std::string::npos)
            layers_str = json.substr(as, ae-as+1);
    }
    if (!layers_str.empty()) {
        size_t p = 0;
        while (true) {
            auto ob = layers_str.find('{', p); if (ob==std::string::npos) break;
            int depth = 0;
            for (size_t i = ob; i < layers_str.size(); i++) {
                if (layers_str[i]=='{') depth++;
                else if (layers_str[i]=='}') { depth--; if (depth==0) {
                    auto obj = layers_str.substr(ob, i-ob+1);
                    LayerConfig lc;
                    auto gt = [&](const std::string& kk){
                        auto pp=obj.find("\""+kk+"\"");
                        if(pp==std::string::npos)return std::string();
                        pp=obj.find(':',pp+kk.size()+2);pp++;
                        while(obj[pp]==' '||obj[pp]=='\t')pp++;
                        if(obj[pp]=='"'){auto ee=obj.find('"',pp+1);return obj.substr(pp+1,ee-pp-1);}
                        auto ee=pp;while(ee<obj.size()&&obj[ee]!=','&&obj[ee]!='}')ee++;
                        return obj.substr(pp,ee-pp);
                    };
                    lc.type = gt("type");
                    lc.layer_type = (lc.type.find("ssm")!=std::string::npos)?0:1;
                    cfg.layers.push_back(lc);
                    p = i+1; break;
                }}
            }
        }
    }

    TensorMap tensors_local;

    // Read tensor index
    fseek(f, idx_off, SEEK_SET);
    std::vector<uint8_t> idx(idx_sz); fread(idx.data(), 1, idx_sz, f);
    size_t pos = 0; std::vector<RinnTensorSpec> specs;
    while (pos < idx_sz) {
        RinnTensorSpec ts;
        int nl = idx[pos] | (idx[pos+1]<<8); pos += 2;
        ts.name.assign((const char*)idx.data()+pos, nl); pos += nl;
        for (int i=0;i<4;i++) { ts.shape[i]=read_int32(idx.data()+pos); pos+=4; }
        ts.quant_type = idx[pos]; pos++;
        int block_size = idx[pos]|(idx[pos+1]<<8); pos+=2;
        memcpy(&ts.offset, idx.data()+pos, 8); pos+=8;
        memcpy(&ts.size, idx.data()+pos, 8); pos+=8;
        
        int nd=4; while(nd>0 && ts.shape[nd-1]==0) nd--;
        ts.n_dim = nd;
        specs.push_back(ts);
    }
    fclose(f);
    
    // Load weights
    FILE* f2 = fopen(path, "rb");
    for (auto& ts : specs) {
        WeightTensor wt;
        wt.quant_type = static_cast<QuantType>(ts.quant_type);
        wt.n_dim = ts.n_dim;
        for (int i = 0; i < wt.n_dim; i++) wt.shape[i] = ts.shape[i];
        wt.n_elems = 1;
        for (int i = 0; i < wt.n_dim; i++) wt.n_elems *= wt.shape[i];

        if (wt.quant_type == QuantType::FP32) {
            size_t bytes = wt.n_elems * sizeof(float);
            std::vector<float> buf(wt.n_elems);
            fseek(f2, ts.offset, SEEK_SET);
            fread(buf.data(), 1, bytes, f2);
            cudaMalloc(&wt.data, bytes);
            cudaMemcpy(wt.data, buf.data(), bytes, cudaMemcpyHostToDevice);
        } else if (wt.quant_type == QuantType::Q4_0 || wt.quant_type == QuantType::Q4_0F) {
            size_t bytes = quantized_size(wt.n_elems, wt.quant_type);
            std::vector<uint8_t> buf(bytes);
            fseek(f2, ts.offset, SEEK_SET);
            fread(buf.data(), 1, bytes, f2);
            cudaMalloc(&wt.data, bytes);
            cudaMemcpy(wt.data, buf.data(), bytes, cudaMemcpyHostToDevice);
        } else {
            fprintf(stderr, "Unsupported quant_type %d for %s\n", (int)wt.quant_type, ts.name.c_str());
        }
        tensors_local.add(ts.name, std::move(wt));
    }
    fclose(f2);
    // Copy to passed-in tensors map
    for (auto& [n, wt] : tensors_local.tensors)
        tensors.add(n, std::move(wt));
    return json;
}

ModelConfig parse_config(const char* path) {
    ModelConfig cfg; TensorMap tmp; read_config(path, cfg, tmp); return cfg;
}
void load_weights(const char* path, TensorMap& tensors) {
    ModelConfig cfg; read_config(path, cfg, tensors);
}
bool load_model(const char* path, ModelConfig& cfg, TensorMap& tensors) {
    auto json = read_config(path, cfg, tensors);
    return !json.empty();
}
