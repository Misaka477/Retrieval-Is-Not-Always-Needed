#include "loader/gguf_reader.h"
#include <cstdio>
#include <cstring>
#include <fstream>

// ── GGUF file parser ───────────────────────────────────────────

bool read_gguf_header(const std::string& path, GGUFMetadata& meta) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }

    uint32_t magic; f.read((char*)&magic, 4);
    if (magic != GGUF_MAGIC) { fprintf(stderr, "Bad magic: 0x%x\n", magic); return false; }
    uint32_t version; f.read((char*)&version, 4);
    if (version < 2 || version > 3) { fprintf(stderr, "Unsupported GGUF version: %u\n", version); return false; }

    uint64_t n_tensors, n_kv;
    f.read((char*)&n_tensors, 8);
    f.read((char*)&n_kv, 8);

    fprintf(stderr, "  parsing %llu metadata KV pairs...\n", (unsigned long long)n_kv);
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t key_len; f.read((char*)&key_len, 8);
        if (key_len > 10000) { fprintf(stderr,"bad key len %llu\n",(unsigned long long)key_len); return false; }
        std::string key(key_len, '\0'); f.read(&key[0], key_len);
        uint32_t val_type; f.read((char*)&val_type, 4);

        auto read_str = [&]() {
            uint64_t len; f.read((char*)&len, 8);
            if (len > 1000000) { f.seekg(len, std::ios::cur); return std::string(); }
            std::string s(len, '\0');
            if (len > 0) f.read(&s[0], len);
            return s;
        };
        auto skip_arr = [&](uint32_t arr_type, uint64_t arr_len) {
            int elem_size = 4;
            if (arr_type == GGUF_TYPE_STRING) elem_size = -1;
            else if (arr_type == GGUF_TYPE_UINT8 || arr_type == GGUF_TYPE_INT8) elem_size = 1;
            else if (arr_type == GGUF_TYPE_UINT16 || arr_type == GGUF_TYPE_INT16) elem_size = 2;
            else if (arr_type == GGUF_TYPE_UINT32 || arr_type == GGUF_TYPE_INT32 || arr_type == GGUF_TYPE_FLOAT32) elem_size = 4;
            else if (arr_type == GGUF_TYPE_UINT64 || arr_type == GGUF_TYPE_INT64 || arr_type == GGUF_TYPE_FLOAT64) elem_size = 8;
            else if (arr_type == GGUF_TYPE_BOOL) elem_size = 1;
            if (elem_size < 0) {
                for (uint64_t j = 0; j < arr_len; j++) {
                    uint64_t sl; f.read((char*)&sl, 8);
                    f.seekg(sl, std::ios::cur);
                }
            } else {
                f.seekg((std::streamoff)((uint64_t)elem_size * arr_len), std::ios::cur);
            }
        };

        if (key == "general.architecture") {
            meta.arch_string = read_str();
            fprintf(stderr, "  architecture=%s\n", meta.arch_string.c_str());
        } else if (val_type == GGUF_TYPE_STRING) {
            read_str();
        } else if (val_type == GGUF_TYPE_FLOAT32) {
            float val; f.read((char*)&val, 4);
            if (key.find("rope.scaling") != std::string::npos && key.find("factor") != std::string::npos)
                meta.rope_scaling_factor = val;
            if (key.find("freq_base") != std::string::npos)
                meta.rope_freq_base = val;
        } else if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32) {
            uint32_t val; f.read((char*)&val, 4);
            auto dot = key.rfind('.');
            auto suff = (dot != std::string::npos) ? key.substr(dot + 1) : key;
            if (suff == "block_count") { meta.n_layers = val; fprintf(stderr,"  n_layers=%d\n",meta.n_layers); }
            if (suff == "embedding_length") { meta.dim = val; fprintf(stderr,"  dim=%d\n",meta.dim); }
            if (suff == "head_count") { meta.n_heads = val; }
            if (suff == "head_count_kv") { meta.n_kv_heads = val; }
            if (suff == "dimension_count") { meta.d_h_r = val; fprintf(stderr,"  rope.dim=%d\n",meta.d_h_r); }
            if (suff == "context_length") { meta.max_seq_len = val; }
            if (suff == "vocab_size") { meta.vocab_size = val; }
            if (suff == "kv_lora_rank") { meta.d_c = val; fprintf(stderr,"  kv_lora_rank=%d\n",meta.d_c); }
            if (suff == "key_length") { meta.dq = val; fprintf(stderr,"  key_length=%d\n",meta.dq); }
            if (suff == "value_length") { meta.v_head_dim = val; fprintf(stderr,"  value_length=%d\n",meta.v_head_dim); }
        } else if (val_type == GGUF_TYPE_ARRAY) {
            uint32_t arr_type; f.read((char*)&arr_type, 4);
            uint64_t arr_len; f.read((char*)&arr_len, 8);
            skip_arr(arr_type, arr_len);
        } else {
            int skip = 0;
            if (val_type == GGUF_TYPE_UINT64 || val_type == GGUF_TYPE_INT64 || val_type == GGUF_TYPE_FLOAT64) skip = 8;
            else if (val_type == GGUF_TYPE_FLOAT32) skip = 4;
            else if (val_type == GGUF_TYPE_BOOL) skip = 1;
            f.seekg(skip, std::ios::cur);
        }
    }

    uint64_t tensor_info_start = (uint64_t)f.tellg();
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t name_len; f.read((char*)&name_len, 8);
        f.seekg(name_len, std::ios::cur);
        uint32_t n_dim; f.read((char*)&n_dim, 4);
        f.seekg((std::streamoff)n_dim * 8, std::ios::cur);
        f.seekg(4 + 8, std::ios::cur);
    }
    uint64_t alignment = 32;
    meta.data_start = (uint64_t)f.tellg();
    meta.data_start = (meta.data_start + alignment - 1) / alignment * alignment;

    f.seekg(tensor_info_start, std::ios::beg);
    meta.tensors.clear();
    meta.tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorInfo ti;
        uint64_t name_len; f.read((char*)&name_len, 8);
        ti.name.resize(name_len); f.read(&ti.name[0], name_len);
        uint32_t n_dim; f.read((char*)&n_dim, 4);
        ti.shape.resize(n_dim);
        for (int j = 0; j < (int)n_dim; j++) {
            uint64_t d; f.read((char*)&d, 8);
            ti.shape[j] = (int64_t)d;
        }
        uint32_t gtype; f.read((char*)&gtype, 4);
        ti.ggml_type = gtype;
        f.read((char*)&ti.offset, 8);
        ti.offset += meta.data_start;
        meta.tensors.push_back(ti);
    }
    return true;
}

// ── CPU dequantization helpers ─────────────────────────────────

constexpr int QK_K = 256;

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        uint32_t m = mant;
        uint32_t e = 113;
        if (m == 0) {
            f32 = (sign << 31);
        } else {
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF;
            f32 = (sign << 31) | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0xFF << 23 | (mant << 13);
        if (mant) f32 |= 0x7FFFFF;
    } else {
        f32 = (sign << 31) | (exp + 127 - 15) << 23 | mant << 13;
    }
    float r; memcpy(&r, &f32, 4); return r;
}

static const float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f
};

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void dequant_q4_K(const uint8_t* src, float* dst, int n) {
    const int BLK = 256;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 144;
        float d = half_to_float(*(const uint16_t*)(blk));
        float min = half_to_float(*(const uint16_t*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;
        int is = 0;
        for (int j = 0; j < BLK && b * BLK + j < n; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;
            int off = b * BLK + j;
            for (int l = 0; l < 32 && off + l < n; l++)
                dst[off + l]      = d1 * (qs[l] & 0xF) - m1;
            for (int l = 0; l < 32 && off + 32 + l < n; l++)
                dst[off + 32 + l] = d2 * (qs[l] >> 4) - m2;
            qs += 32; is += 2;
        }
    }
}

void dequant_q5_K(const uint8_t* src, float* dst, int n) {
    const int BLK = 256;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 176;
        float d = half_to_float(*(const uint16_t*)(blk));
        float min = half_to_float(*(const uint16_t*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* ql = blk + 48;
        int is = 0;
        for (int j = 0; j < BLK && b * BLK + j < n; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;
            int off = b * BLK + j;
            for (int l = 0; l < 32 && off + l < n; l++) {
                int ql_l = ql[l] & 0xF;
                int qh_bit = (qh[l/2] >> ((l % 2) * 4)) & 1;
                dst[off + l] = d1 * ((ql_l | (qh_bit << 4))) - m1;
            }
            for (int l = 0; l < 32 && off + 32 + l < n; l++) {
                int ql_l = ql[l] >> 4;
                int qh_bit = (qh[l/2] >> ((l % 2) * 4 + 4)) & 1;
                dst[off + 32 + l] = d2 * ((ql_l | (qh_bit << 4))) - m2;
            }
            ql += 32; qh += 16; is += 2;
        }
    }
}

void dequant_q6_K(const uint8_t* src, float* dst, int n) {
    const int BLK = 256;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 210;
        float d = half_to_float(*(const uint16_t*)(blk + 208));
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = (const int8_t*)(blk + 192);
        for (int nblk = 0; nblk < BLK && b * BLK + nblk < n; nblk += 128) {
            int off = b * BLK + nblk;
            for (int l = 0; l < 32 && off + l < n; l++) {
                int is = l / 16;
                int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[off + l]       = d * sc[is + 0] * q1;
                dst[off + l + 32]  = d * sc[is + 2] * q2;
                dst[off + l + 64]  = d * sc[is + 4] * q3;
                dst[off + l + 96]  = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32;
        }
    }
}

void dequant_iq4_xs(const uint8_t* src, float* dst, int n) {
    const int BLK = 256;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 136;
        float d = half_to_float(*(const uint16_t*)(blk));
        uint16_t scales_h = *(const uint16_t*)(blk + 2);
        const uint8_t* scales_l = blk + 4;
        const uint8_t* qs = blk + 8;
        int off = b * BLK;
        for (int ib = 0; ib < BLK/32 && off + ib * 32 < n; ib++) {
            int ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xf;
            ls |= ((scales_h >> 2*ib) & 3) << 4;
            float dl = d * (ls - 32);
            for (int j = 0; j < 16 && off + ib * 32 + j < n; j++)
                dst[off + ib * 32 + j]      = dl * kvalues_iq4nl[qs[j] & 0xf];
            for (int j = 0; j < 16 && off + ib * 32 + 16 + j < n; j++)
                dst[off + ib * 32 + 16 + j] = dl * kvalues_iq4nl[qs[j] >> 4];
            qs += 16;
        }
    }
}

void dequant_block_f16(const uint8_t* src, float* dst, int n) {
    const uint16_t* h = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        uint32_t sign = (h[i] >> 15) & 1;
        uint32_t exp = (h[i] >> 10) & 0x1F;
        uint32_t mant = h[i] & 0x3FF;
        uint32_t f32;
        if (exp == 0) {
            f32 = (sign << 31) | (0x7F - 15 + 1) << 23 | mant << 13;
        } else if (exp == 31) {
            f32 = (sign << 31) | 0xFF << 23 | (mant ? 0x7FFFFF : 0);
        } else {
            f32 = (sign << 31) | (exp + 127 - 15) << 23 | mant << 13;
        }
        memcpy(dst + i, &f32, 4);
    }
}

void dequant_q4_0(const uint8_t* src, float* dst, int n) {
    const int BLK = 32;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        uint16_t scale_half; memcpy(&scale_half, src + b * 18, 2);
        int sign = (scale_half >> 15) & 1;
        int exp = (scale_half >> 10) & 0x1F;
        int mant = scale_half & 0x3FF;
        float scale;
        uint32_t f32 = (sign << 31) | ((exp == 0 ? 0 : exp + 127 - 15) << 23) | (mant << 13);
        if (exp == 0) f32 = (sign << 31) | (0x7F - 15 + 1) << 23 | mant << 13;
        memcpy(&scale, &f32, 4);

        const uint8_t* data = src + b * 18 + 2;
        for (int i = 0; i < 32 && b * 32 + i < n; i++) {
            int q = (data[i / 2] >> ((i & 1) << 2)) & 0xF;
            dst[b * 32 + i] = (q - 7) * scale;
        }
    }
}
