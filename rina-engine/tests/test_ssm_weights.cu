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
    
    // Verify w_dq weight directly from TensorMap
    auto* t = w.get("transformer.h.0.path.w_dq.weight");
    if(!t){fprintf(stderr,"not found\n");return 1;}
    
    printf("w_dq weight: ptr=%p, shape[0]=%d shape[1]=%d n_elems=%d\n",
           t->data, t->shape[0], t->shape[1], t->n_elems);
    
    // Dump first 10 values
    std::vector<float> cpu(10);
    cudaMemcpy(cpu.data(), t->data, 10*sizeof(float), cudaMemcpyDeviceToHost);
    printf("First 10 values: ");
    for(int i=0;i<10;i++) printf("%.6f ", cpu[i]);
    printf("\n");
    
    // Dump element at row 0, col 0-9
    int K = t->shape[1]; // 640
    std::vector<float> cpu_row0(10);
    cudaMemcpy(cpu_row0.data(), t->data, 10*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Row 0, cols 0-9: ");
    for(int i=0;i<10;i++) printf("%.6f ", cpu_row0[i]);
    printf("\n");
    
    // The weight is [160, 640]. Row 0 has 640 elements.
    // Row 1 starts at offset 640.
    std::vector<float> cpu_row1(10);
    cudaMemcpy(cpu_row1.data(), (const float*)t->data + 640, 10*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Row 1, cols 0-9: ");
    for(int i=0;i<10;i++) printf("%.6f ", cpu_row1[i]);
    printf("\n");
    
    w.free_all();
    return 0;
}
