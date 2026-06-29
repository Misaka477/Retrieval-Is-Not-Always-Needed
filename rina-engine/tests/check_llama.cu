#include <cuda_runtime.h>
#include <cstdio>
#include "core/config.h"
#include "core/tensor.h"
#include "model.h"
int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/llama3.2-1b.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    printf("cfg.name=%s\n", cfg.name.c_str());
    auto q = w.get("transformer.h.0.attn.rope_q.cos");
    auto k = w.get("transformer.h.0.attn.rope.cos");
    auto wte = w.get("transformer.wte.weight");
    auto ln1 = w.get("transformer.h.0.ln1.weight");
    printf("rope_q.cos: %p (shape=%dx%d, size=%d)\n", q?q->data:0, q?q->shape[0]:0, q?q->shape[1]:0, q?q->n_elems:0);
    printf("rope.cos:   %p (shape=%dx%d, size=%d)\n", k?k->data:0, k?k->shape[0]:0, k?k->shape[1]:0, k?k->n_elems:0);
    printf("wte:        %p\n", wte?wte->data:0);
    printf("ln1.weight: %p (n_elems=%d)\n", ln1?ln1->data:0, ln1?ln1->n_elems:0);
    w.free_all();
    return 0;
}
