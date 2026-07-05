// model_fp32.cu — legacy v1 forward, now delegates to v2 (Layer-based)
#include "model.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include <cstdio>

void model_forward_fp32(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream,
    int start_pos) {
    model_forward_v2(cfg, w, ids, logits, B, T, stream, start_pos);
}
