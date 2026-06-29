#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

// Forward declarations
struct GradBuffers;

// A recorded operation for backward replay
struct OpRecord {
    const char* name;
    const float* inputs[4];
    float* outputs[4];
    const float* weights[2];
    size_t wg_offsets[2];
    int args[8];
    void (*backward_fn)(const OpRecord&, GradBuffers&, float* /* wg */, cudaStream_t);
};

// Simple tape: records forward ops, replays backward in reverse order
struct Tape {
    std::vector<OpRecord> ops;

    void record(OpRecord&& op);
    void replay(GradBuffers& grad, float* wg, cudaStream_t stream);
    void clear();
};
