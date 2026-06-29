#pragma once
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// Sparse Index Manager — builds attention mask using cq similarity
// Mirrors model_l3x.py SparseIndexManager
struct SparseIndexManager {
    int window, k, local_w;
    int step = 0, last_T = 0;
    std::vector<uint8_t> mask_data; // [T, T] boolean
    int current_T = 0;

    SparseIndexManager(int window_=32, int k_=8, int local_w_=4)
        : window(window_), k(k_), local_w(local_w_) {}

    void reset() { step = 0; last_T = 0; mask_data.clear(); current_T = 0; }

    bool should_recompute() const { return mask_data.empty() || step % window == 0; }

    // Normalize vectors in-place and return norms
    static void normalize_2d(float* data, int rows, int cols) {
        for (int r = 0; r < rows; r++) {
            float sum = 0;
            for (int c = 0; c < cols; c++) sum += data[r*cols+c] * data[r*cols+c];
            float inv = 1.0f / (std::sqrt(sum) + 1e-8f);
            for (int c = 0; c < cols; c++) data[r*cols+c] *= inv;
        }
    }

    void recompute(const float* cq, int T) {
        // cq: [B, T, d_c], use batch 0 only
        std::vector<float> cq_norm(T * (cq ? 160 : 0)); // d_c=160
        int d_c = cq ? 160 : 0;
        // Copy and normalize
        if (cq) {
            std::memcpy(cq_norm.data(), cq, T * d_c * sizeof(float));
            normalize_2d(cq_norm.data(), T, d_c);
        }

        // Compute similarity matrix: sim[i,j] = cq_norm[i] · cq_norm[j]
        std::vector<float> sim(T * T, 0);
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
                for (int c = 0; c < d_c; c++)
                    sim[i*T+j] += cq_norm[i*d_c+c] * cq_norm[j*d_c+c];

        // Top-k indices per row
        std::vector<int> topk_idx(T * std::min(k, T-1));
        for (int i = 0; i < T; i++) {
            int nk = std::min(k, T-1);
            // We need the top-nk values. Use simple selection.
            std::vector<std::pair<float,int>> sorter;
            for (int j = 0; j < T; j++) if (j != i)
                sorter.push_back({sim[i*T+j], j});
            std::sort(sorter.begin(), sorter.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
            for (int j = 0; j < nk; j++)
                topk_idx[i*nk + j] = sorter[j].second;
        }

        // Build mask
        mask_data.resize(T * T, 0);
        current_T = T;
        for (int i = 0; i < T; i++) {
            int ls = std::max(0, i - local_w);
            for (int s = ls; s <= i; s++)
                mask_data[i*T + s] = 1; // local window
            if (i > 0) {
                int nk = std::min(k, T-1);
                for (int j = 0; j < nk; j++)
                    mask_data[i*T + topk_idx[(i-1)*nk + j]] = 1; // top-k from prev cq
            }
        }
        // Causal mask: only allow s ≤ i
        for (int i = 0; i < T; i++)
            for (int s = i+1; s < T; s++)
                mask_data[i*T + s] = 0;
        last_T = T;
    }

    void extend(int T, const float* cq) {
        int old_T = last_T;
        std::vector<uint8_t> new_mask(T * T, 0);
        // Copy old mask
        for (int i = 0; i < old_T; i++)
            for (int j = 0; j < old_T; j++)
                new_mask[i*T + j] = mask_data[i*old_T + j];
        // New positions
        int d_c = cq ? 160 : 0;
        for (int i = old_T; i < T; i++) {
            int ls = std::max(0, i - local_w);
            for (int s = ls; s <= i; s++)
                new_mask[i*T + s] = 1;
            if (cq && i > 0) {
                // cq_i · cq_past where cq_i is [1,1,d_c] and cq_past is [1,i,d_c]
                std::vector<float> cq_i_norm(d_c), cq_past_norm(i * d_c);
                std::memcpy(cq_i_norm.data(), cq + i*d_c, d_c*sizeof(float));
                std::memcpy(cq_past_norm.data(), cq, i*d_c*sizeof(float));
                normalize_2d(cq_i_norm.data(), 1, d_c);
                normalize_2d(cq_past_norm.data(), i, d_c);
                std::vector<float> sim_local(i);
                for (int j = 0; j < i; j++)
                    for (int c = 0; c < d_c; c++)
                        sim_local[j] += cq_i_norm[c] * cq_past_norm[j*d_c+c];
                // Top-k
                int nk = std::min(k, i);
                std::vector<std::pair<float,int>> sorter;
                for (int j = 0; j < i; j++)
                    sorter.push_back({sim_local[j], j});
                std::sort(sorter.begin(), sorter.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
                for (int j = 0; j < nk; j++)
                    new_mask[i*T + sorter[j].second] = 1;
            }
            new_mask[i*T + i] = 1;
        }
        mask_data.swap(new_mask);
        current_T = T;
        last_T = T;
    }

    void step_forward(const float* cq, int T) {
        if (should_recompute())
            recompute(cq, T);
        else if (T > last_T)
            extend(T, cq);
        step++;
    }

    // Returns attention mask: 0.0 where allowed, -inf where masked
    void get_mask_attn(float* out, int T) const {
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
                out[i*T + j] = mask_data[i*T+j] ? 0.0f : -INFINITY;
    }

    int get_K_use(int T) const {
        int max_k = 0;
        for (int i = 0; i < T; i++) {
            int cnt = 0;
            for (int j = 0; j < T; j++) if (mask_data[i*T+j]) cnt++;
            if (cnt > max_k) max_k = cnt;
        }
        return max_k;
    }
};

// Build gather indices from mask.
// mask[T,T] = 1 where allowed. Returns idx[T, K_use] and K_use.
static void build_gather_indices(const uint8_t* mask, int* idx_out, int& K_use, int T) {
    int max_k = 0;
    for (int i = 0; i < T; i++) {
        int cnt = 0;
        for (int j = 0; j < T; j++) if (mask[i*T+j]) cnt++;
        if (cnt > max_k) max_k = cnt;
    }
    K_use = max_k;
    for (int i = 0; i < T; i++) {
        int pos = 0;
        for (int j = 0; j < T; j++) {
            if (mask[i*T+j]) {
                if (pos < K_use) idx_out[i*K_use + pos] = j;
                pos++;
            }
        }
        // Pad with last valid index
        for (; pos < K_use; pos++) idx_out[i*K_use + pos] = idx_out[i*K_use + (pos-1 > 0 ? pos-1 : 0)];
    }
}
