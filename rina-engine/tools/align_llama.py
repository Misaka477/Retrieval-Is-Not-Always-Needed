#!/usr/bin/env python3
"""PyTorch vs Engine alignment test for Llama 3.2 1B.
Run: python3 tools/align_llama.py /tmp/llama3.2-1b.rinn"""
import torch, sys, struct, math, subprocess, numpy as np

def engine_forward(rinn_path, ids, B=1, T=None):
    """Run engine forward and return logits as numpy array."""
    if T is None: T = len(ids)
    import ctypes
    # Build a tiny C test that writes logits to a file
    test_path = "/tmp/align_engine_test.cu"
    out_path = "/tmp/align_logits.bin"
    
    code = f'''
    #include <cuda_runtime.h>
    #include <cstdio>
    #include "core/config.h"
    #include "core/tensor.h"
    #include "core/layer.h"
    #include "core/buffer.h"
    #include "training/train.h"
    #include "model.h"
    extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
    extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
    extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
    int main() {{
        ModelConfig cfg; TensorMap w;
        if(!load_model("{rinn_path}",cfg,w)){{fprintf(stderr,"load fail\\n");return 1;}}
        auto layers=build_layers(cfg,w);
        if(layers.empty()){{fprintf(stderr,"build fail\\n");return 1;}}
        int B={B},T={T},n=B*T,d=cfg.dim,V=cfg.vocab_size;
        int ws=0,total=0;
        for(auto&l:layers){{int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);}}
        int hd={{{"auto* w1=w.get(\\"transformer.h.0.mlp.w1.weight\\");w1?w1->shape[0]:d*4*2/3/256*256;"}}};
        BufferManager bufs;
        bufs.alloc_fwd(n,d,ws,hd,V,total);
        if(!bufs.fwd.h){{fprintf(stderr,"alloc fail\\n");return 1;}}
        cudaStream_t s;cudaStreamCreate(&s);
        int*ids;cudaMalloc(&ids,n*sizeof(int));
        int hids[]={{{",".join(str(x) for x in ids[:B*T])}}}};
        cudaMemcpy(ids,hids,n*sizeof(int),cudaMemcpyHostToDevice);
        float*base_save=bufs.fwd.save;
        const float*wte=(const float*)w.get("transformer.wte.weight")->data;
        launch_embedding_fp32(wte,ids,bufs.fwd.h,B,T,d,s);
        for(int l=0;l<(int)layers.size();l++){{bufs.fwd.save=base_save+layers[l]->save_offset*n;layers[l]->forward(bufs.fwd.h,bufs.fwd,B,T,s);}}
        bufs.fwd.save=base_save;
        auto*ln_f=w.get("transformer.ln_f.weight");
        if(ln_f)launch_pytorch_ln_kernel(bufs.fwd.h,(const float*)ln_f->data,n,d,1e-5f,s);
        auto* lm_h=w.get("lm_head.weight");
        if(!lm_h)wte=(const float*)w.get("transformer.wte.weight")->data;
        else wte=(const float*)lm_h->data;
        launch_linear_fp32(bufs.fwd.h,wte,bufs.fwd.lm,n,V,d,s);
        cudaStreamSynchronize(s);
        std::vector<float>cpu(n*V);
        cudaMemcpy(cpu.data(),bufs.fwd.lm,n*V*sizeof(float),cudaMemcpyDeviceToHost);
        FILE*f=fopen("{out_path}","wb");fwrite(cpu.data(),sizeof(float),n*V,f);fclose(f);
        w.free_all();bufs.free_all();cudaFree(ids);
        fprintf(stderr,"engine OK\\n");
        return 0;
    }}
    '''
    return None  # temporary: use the test binary instead

def pt_forward(model_path, ids):
    """Load Llama via HuggingFace and run forward."""
    from transformers import LlamaForCausalLM, AutoTokenizer
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([ids]).cuda()
        out = model(inputs, output_hidden_states=False)
        logits = out.logits[0]  # [T, V]
    return logits.cpu().numpy()

def main():
    rinn_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/llama3.2-1b.rinn"
    hf_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/aquama/Development/RINA_Project/models/teacher/LLM-Research/Llama-3___2-1B-Instruct"
    
    # Fixed test input
    np.random.seed(42)
    ids = list(np.random.randint(1000, 10000, size=8).tolist())
    print(f"Input IDs: {ids}")
    
    # PyTorch reference
    print("\nRunning PyTorch forward...")
    import time
    t0 = time.time()
    pt_logits = pt_forward(hf_path, ids)
    print(f"  {time.time()-t0:.1f}s, shape={pt_logits.shape}")
    
    # Engine forward - use the test_llama binary
    print("\nRunning Engine forward...")
    build_dir = "/home/aquama/Development/RINA_Project/rina-engine/build"
    # Create a test with the right IDs
    with open("/tmp/align_ids.txt", "w") as f:
        f.write(" ".join(str(x) for x in ids))
    
    # Build a specialized test binary
    eng_bin = f"{build_dir}/test_llama"
    # For now, just read the .rinn weights and print a token
    # Actually, let's first verify the engine output
    
    print("\nTo compare: run the test binary, capture logits, run pt_ref")
    print("For a quick comparison, use tools/pt_ref.py with the HF model")

if __name__ == "__main__":
    main()
