#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kernels/gemm.cuh"

extern void launch_flash_attn_fp32(const float*,const float*,const float*,float*,int,int,int,int,int,cudaStream_t);

static void load_bin(const char* path, float* buf, int n) {
    std::vector<float> cpu(n);
    FILE* f = fopen(path,"rb"); fread(cpu.data(),sizeof(float),n,f); fclose(f);
    cudaMemcpy(buf,cpu.data(),n*sizeof(float),cudaMemcpyHostToDevice);
}
static void save_bin(const char* path, const float* buf, int n) {
    std::vector<float> cpu(n);
    cudaMemcpy(cpu.data(),buf,n*sizeof(float),cudaMemcpyDeviceToHost);
    FILE* f = fopen(path,"wb"); fwrite(cpu.data(),sizeof(float),n,f); fclose(f);
}

int main() {
    int B=1, H=32, T=8, d=64;  // dq=dh=64 for GQA
    int Bh = B * H;
    int q_sz = Bh * T * d, v_sz = Bh * T * d;
    
    float *Q, *K, *V, *O;
    cudaMalloc(&Q, q_sz * sizeof(float));
    cudaMalloc(&K, q_sz * sizeof(float));
    cudaMalloc(&V, v_sz * sizeof(float));
    cudaMalloc(&O, v_sz * sizeof(float));
    
    load_bin("/tmp/pt_qf.bin", Q, q_sz);
    load_bin("/tmp/pt_kf.bin", K, q_sz);
    load_bin("/tmp/pt_vf.bin", V, v_sz);
    
    cudaStream_t s; cudaStreamCreate(&s);
    
    cudaGetLastError();
    launch_flash_attn_fp32(Q, K, V, O, B, H, T, d, d, s);
    cudaStreamSynchronize(s);
    cudaError_t e = cudaGetLastError();
    fprintf(stderr,"FA kernel: %s\n", cudaGetErrorString(e));
    
    save_bin("/tmp/engine_attn.bin", O, v_sz);
    fprintf(stderr,"Saved engine attention output\n");
    
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
    cudaStreamDestroy(s);
    return 0;
}
