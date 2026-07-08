cmake_minimum_required(VERSION 3.22)
project(rina-engine LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "86-real")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

set(TORCH_ROOT "/home/aquama/miniconda3/envs/natalia/lib/python3.12/site-packages/torch" CACHE PATH "Path to PyTorch")
if(EXISTS ${TORCH_ROOT})
    add_definitions(-DRINA_WITH_PYTORCH)
    include_directories(${TORCH_ROOT}/include)
    include_directories(${TORCH_ROOT}/include/torch/csrc/api/include)
    link_directories(${TORCH_ROOT}/lib)
endif()

set(RINA_SRC src/loader.cpp src/main.cpp src/core/layer.cpp src/core/buffer.cpp
    src/core/tape.cpp src/core/layers_fp32.cpp src/core/tokenizer.cpp
    src/loader_hf.cpp src/loader_gguf.cpp
    src/loader/gguf_reader.cpp src/loader/loader_shared.cpp
    src/loader/loader_gqa.cpp src/loader/loader_mla.cpp
    src/infer/infer_shared.cpp src/infer/infer_gqa.cpp src/infer/infer_mla.cpp)
if(EXISTS ${TORCH_ROOT})
    list(APPEND RINA_SRC src/model_pytorch.cpp)
endif()

set(RINA_CUDA_SRC
    src/ops/linear.cu src/ops/rms_norm.cu src/ops/rope.cu
    src/ops/embedding.cu src/ops/silu_mul.cu src/ops/saxpy.cu
    src/kernels/pytorch_ln_kernel.cu
    src/kernels/build_qkv_fp32.cu src/kernels/flash_fp32.cu
    src/kernels/quantize.cu src/kernels/sparse_attn.cu src/kernels/linear_fp32.cu
    src/kernels/linear_q4k.cu src/kernels/linear_q6k.cu src/kernels/linear_iq4xs.cu
    src/kernels/linear_all_quant.cu
    src/kernels/linear_q4k.cu src/kernels/linear_q6k.cu src/kernels/linear_iq4xs.cu
    src/kernels/ssm_kernels_fp32.cu src/kernels/ssm_lsc_scan.cu src/kernels/activation_fp32.cu
    src/kernels/loss_fp32.cu src/kernels/norm_fp32.cu
    src/model_fp32.cu src/model_forward_v2.cu src/model_train_v2.cu
    src/kernels/optimizer_fp32.cu src/kernels/tensor_utils.cu src/train_fp32.cu
    src/arch/gqa/gqa_layer.cu src/arch/ssm/rina_ssm_layer.cu src/arch/mla/rina_mla_layer.cu
    src/arch/mla/deepseek_mla_layer.cu src/arch/register_layers.cu src/training/train.cu
    # llama.cpp fused quantized matmul (MMQ)
    src/llama_kernels/rina_backend.cu
    src/llama_kernels/rina_mmq_wrapper.cu
    src/llama_kernels/mmq.cu
    src/llama_kernels/mmid.cu
    src/llama_kernels/quantize.cu
    src/llama_kernels/template-instances/mmq-instance-q4_0.cu
    src/llama_kernels/template-instances/mmq-instance-q4_1.cu
    src/llama_kernels/template-instances/mmq-instance-q5_0.cu
    src/llama_kernels/template-instances/mmq-instance-q5_1.cu
    src/llama_kernels/template-instances/mmq-instance-q8_0.cu
    src/llama_kernels/template-instances/mmq-instance-q2_k.cu
    src/llama_kernels/template-instances/mmq-instance-q3_k.cu
    src/llama_kernels/template-instances/mmq-instance-q4_k.cu
    src/llama_kernels/template-instances/mmq-instance-q5_k.cu
    src/llama_kernels/template-instances/mmq-instance-q6_k.cu
    src/llama_kernels/template-instances/mmq-instance-iq2_xxs.cu
    src/llama_kernels/template-instances/mmq-instance-iq2_xs.cu
    src/llama_kernels/template-instances/mmq-instance-iq2_s.cu
    src/llama_kernels/template-instances/mmq-instance-iq3_xxs.cu
    src/llama_kernels/template-instances/mmq-instance-iq3_s.cu
    src/llama_kernels/template-instances/mmq-instance-iq1_s.cu
    src/llama_kernels/template-instances/mmq-instance-iq4_nl.cu
    src/llama_kernels/template-instances/mmq-instance-iq4_xs.cu
    src/llama_kernels/template-instances/mmq-instance-mxfp4.cu
    src/llama_kernels/template-instances/mmq-instance-nvfp4.cu
    src/llama_kernels/template-instances/mmq-instance-q1_0.cu)

set(RINA_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/llama.cpp")
add_executable(rina_infer ${RINA_SRC} ${RINA_CUDA_SRC})
target_include_directories(rina_infer PUBLIC src)
# llama.cpp kernel headers (private to rina_infer only)
target_include_directories(rina_infer PRIVATE src/llama_kernels)
target_include_directories(rina_infer PRIVATE ${RINA_LLAMA_DIR}/ggml/include)
target_include_directories(rina_infer PRIVATE ${RINA_LLAMA_DIR}/ggml/src)
target_include_directories(rina_infer PRIVATE ${RINA_LLAMA_DIR}/ggml/src/ggml-cuda)
set(RINA_COMMON_LIBS CUDA::cudart CUDA::cublas CUDA::cuda_driver dl m stdc++fs)
target_link_libraries(rina_infer PRIVATE ${RINA_COMMON_LIBS})

set(RINA_SRC_NO_MAIN src/loader.cpp src/core/layer.cpp src/core/buffer.cpp
    src/core/tape.cpp src/core/layers_fp32.cpp src/core/tokenizer.cpp
    src/loader_hf.cpp src/loader_gguf.cpp
    src/loader/gguf_reader.cpp src/loader/loader_shared.cpp
    src/loader/loader_gqa.cpp src/loader/loader_mla.cpp
    src/infer/infer_shared.cpp src/infer/infer_gqa.cpp src/infer/infer_mla.cpp)

add_executable(eval_ppl ${RINA_SRC_NO_MAIN} ${RINA_CUDA_SRC} eval/eval_ppl.cpp)
target_include_directories(eval_ppl PUBLIC src)
target_include_directories(eval_ppl PRIVATE src/llama_kernels)
target_include_directories(eval_ppl PRIVATE ${RINA_LLAMA_DIR}/ggml/include)
target_include_directories(eval_ppl PRIVATE ${RINA_LLAMA_DIR}/ggml/src)
target_include_directories(eval_ppl PRIVATE ${RINA_LLAMA_DIR}/ggml/src/ggml-cuda)
target_link_libraries(eval_ppl PRIVATE ${RINA_COMMON_LIBS})

add_executable(kernel_compare ${RINA_SRC_NO_MAIN} ${RINA_CUDA_SRC} tools/kernel_compare.cpp)
target_include_directories(kernel_compare PUBLIC src)
target_link_libraries(kernel_compare PRIVATE ${RINA_COMMON_LIBS})

set(RINA_TEST_CPP src/loader.cpp src/core/layer.cpp src/core/layers_fp32.cpp
    src/core/buffer.cpp src/core/tape.cpp)

macro(add_rina_test name src)
    add_executable(${name} ${RINA_TEST_CPP} ${RINA_CUDA_SRC} ${src})
    target_include_directories(${name} PUBLIC src)
    target_link_libraries(${name} PRIVATE ${RINA_COMMON_LIBS})
endmacro()

add_rina_test(test_train tests/test_train.cu)
add_rina_test(test_load tests/test_load.cu)
add_rina_test(test_gradcheck tests/test_gradcheck.cu)
add_rina_test(test_align tests/test_align.cu)
add_rina_test(test_v2_train tests/test_v2_train.cu)
add_rina_test(test_ssm_debug tests/test_ssm_debug.cu)
add_rina_test(test_mla_v2 tests/test_mla_v2.cu)
add_rina_test(test_llama tests/test_llama.cu)
add_rina_test(align_llama tests/align_llama.cu)
add_rina_test(align_debug tests/align_debug.cu)
add_rina_test(align_gqa_layer tests/align_gqa_layer.cu)
add_rina_test(fa_test tests/fa_test.cu)

if(EXISTS ${TORCH_ROOT})
    target_link_libraries(rina_infer PRIVATE
        ${TORCH_ROOT}/lib/libtorch.so ${TORCH_ROOT}/lib/libc10.so
        ${TORCH_ROOT}/lib/libtorch_cpu.so ${TORCH_ROOT}/lib/libtorch_cuda.so)
    set_target_properties(rina_infer PROPERTIES
        BUILD_RPATH "${TORCH_ROOT}/lib;${TORCH_ROOT}/../lib" INSTALL_RPATH "${TORCH_ROOT}/lib;${TORCH_ROOT}/../lib")
endif()
add_rina_test(test_infer tests/test_infer.cu)
add_rina_test(test_jamba_align tests/test_jamba_align.cu)
add_rina_test(test_jamba_step tests/test_jamba_step.cu)
add_rina_test(test_linear_direct tests/test_linear_direct.cu)
add_rina_test(test_verify_weights tests/test_verify_weights.cu)
add_rina_test(test_linear_m3 tests/test_linear_m3.cu)
add_rina_test(test_linear_q4 tests/test_linear_q4.cu)
add_rina_test(test_ssm_weights tests/test_ssm_weights.cu)
add_rina_test(test_save_ptr tests/test_save_ptr.cu)
add_rina_test(test_mla_align tests/test_mla_align.cu)
add_rina_test(test_a_align tests/test_a_align.cu)
add_rina_test(test_jamba_mla_step tests/test_jamba_mla_step.cu)
add_rina_test(test_jamba_fwd_dump tests/test_jamba_fwd_dump.cu)
add_rina_test(test_jamba_compare_fwd tests/test_jamba_compare_fwd.cu)
add_rina_test(test_bf16_align tests/test_bf16_align.cu)


