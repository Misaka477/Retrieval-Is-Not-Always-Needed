#include "core/tape.h"

void Tape::record(OpRecord&& op) {
    ops.push_back(std::move(op));
}

void Tape::replay(GradBuffers& grad, float* wg, cudaStream_t stream) {
    for (int i = (int)ops.size() - 1; i >= 0; i--) {
        if (ops[i].backward_fn)
            ops[i].backward_fn(ops[i], grad, wg, stream);
    }
}

void Tape::clear() {
    ops.clear();
}
