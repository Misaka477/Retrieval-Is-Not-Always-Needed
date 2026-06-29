// train_fp32.cu — legacy v1 training, now delegates to v2 (Layer-based)
#include "model.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include <cstdio>

float model_train(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, const int* targets, float* loss_d,
    int B, int T, int step, cudaStream_t stream) {
    return model_train_v2(cfg, w, ids, targets, loss_d, B, T, step, stream);
}
