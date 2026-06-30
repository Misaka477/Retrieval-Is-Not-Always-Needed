#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "core/config.h"
#include "core/tensor.h"
#include "model.h"

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/jamba_qw2.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    
    auto* wt = w.get("transformer.h.0.path.w_dq.weight");
    if (!wt) { fprintf(stderr, "weight not found\n"); return 1; }
    
    std::vector<float> cpu(wt->n_elems);
    cudaMemcpy(cpu.data(), wt->data, wt->n_elems*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write to file
    FILE* f = fopen("/tmp/eng_wdq_weight.bin", "wb");
    fwrite(cpu.data(), sizeof(float), wt->n_elems, f);
    fclose(f);
    fprintf(stderr, "Dumped w_dq weight: %d elems, shape=[%d,%d]\n", 
            wt->n_elems, wt->shape[0], wt->shape[1]);
    
    w.free_all(); return 0;
}
