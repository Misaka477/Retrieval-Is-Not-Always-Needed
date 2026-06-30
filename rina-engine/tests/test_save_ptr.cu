#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "core/config.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/jamba_qw2.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    auto layers=build_layers(cfg,w);
    
    int B=1,T=3,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int ws=0,total=0;
    for(auto& l:layers){
        int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;
        total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
    }
    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,1536,V,total);
    
    cudaStream_t s;cudaStreamCreate(&s);
    int*d_ids;cudaMalloc(&d_ids,n*sizeof(int));
    int ids[]={128000,9906,11};
    cudaMemcpy(d_ids,ids,n*sizeof(int),cudaMemcpyHostToDevice);
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;
    extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
    launch_embedding_fp32(wte,d_ids,bufs.fwd.h,B,T,d,s);
    cudaStreamSynchronize(s);
    
    float* base_save = bufs.fwd.save;
    printf("base_save = %p, save_offset[0] = %d\n", (void*)base_save, layers[0]->save_offset);
    
    bufs.fwd.save = base_save + layers[0]->save_offset * n;
    printf("bufs.fwd.save (before forward) = %p\n", (void*)bufs.fwd.save);
    
    // Call forward
    layers[0]->forward(bufs.fwd.h, bufs.fwd, B, T, s);
    cudaStreamSynchronize(s);
    
    printf("bufs.fwd.save (after forward) = %p\n", (void*)bufs.fwd.save);
    
    // Now dump the cq portion from the save buffer
    // off_cq = d = 640 for this model
    int off_cq = 640;  // from SSM init: off_cq = d
    float* cq_addr = base_save + layers[0]->save_offset * n + off_cq * n;
    printf("Expected cq addr = %p\n", (void*)cq_addr);
    printf("base_save + off_cq*n = %p\n", (void*)(base_save + off_cq * n));
    
    // Read cq directly from this address
    std::vector<float> cq_direct(n * 160);
    cudaMemcpy(cq_direct.data(), cq_addr, n * 160 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cq direct from computed addr: [%.6f, %.6f, %.6f]\n", 
           cq_direct[0], cq_direct[1], cq_direct[2]);
    
    // Also read from base_save + off_cq*n (as the SSM layer would write to)
    float* ssm_writes_to = base_save + off_cq * n;
    std::vector<float> cq_ssm(160);
    cudaMemcpy(cq_ssm.data(), ssm_writes_to, 160 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cq at base_save + off_cq*n: [%.6f, %.6f, %.6f]\n",
           cq_ssm[0], cq_ssm[1], cq_ssm[2]);
    
    // Read at base_save + save_offset*n + off_cq*n (the full correct address)
    std::vector<float> cq_correct(160);
    cudaMemcpy(cq_correct.data(), base_save + layers[0]->save_offset * n + off_cq * n, 160 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cq at correct addr: [%.6f, %.6f, %.6f]\n",
           cq_correct[0], cq_correct[1], cq_correct[2]);
    
    w.free_all();bufs.free_all();cudaFree(d_ids);cudaStreamDestroy(s);
    return 0;
}
