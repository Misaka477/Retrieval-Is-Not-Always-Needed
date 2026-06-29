cmake_minimum_required(VERSION 3.22)
project(rina-engine LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "86-real")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

set(TORCH_ROOT /home/aquama/miniconda3/envs/natalia/lib/python3.12/site-packages/torch)
if(EXISTS ${TORCH_ROOT})
    add_definitions(-DRINA_WITH_PYTORCH)
    include_directories(${TORCH_ROOT}/include)
    include_directories(${TORCH_ROOT}/include/torch/csrc/api/include)
    link_directories(${TORCH_ROOT}/lib)
endif()

set(RINA_SRC src/loader.cpp src/main.cpp src/core/layer.cpp src/core/buffer.cpp
    src/core/tape.cpp src/core/layers_fp32.cpp
    src/model_pytorch.cpp)

set(RINA_CUDA_SRC
    src/kernels/rope_fp32.cu src/kernels/embedding.cu src/kernels/pytorch_ln_kernel.cu
    src/kernels/rms_norm.cu src/kernels/build_qkv_fp32.cu src/kernels/flash_fp32.cu
    src/kernels/quantize.cu src/kernels/sparse_attn.cu src/kernels/linear_fp32.cu
    src/kernels/ssm_kernels_fp32.cu src/kernels/ssm_lsc_scan.cu src/kernels/activation_fp32.cu
    src/kernels/loss_fp32.cu src/kernels/norm_fp32.cu
    src/model_fp32.cu src/kernels/optimizer_fp32.cu src/train_fp32.cu
    src/model_v2.cu
    src/arch/gqa/gqa_layer.cu src/arch/ssm/ssm_layer.cu src/arch/mla/mla_layer.cu
    src/arch/register_layers.cu src/training/train.cu)

add_executable(rina_infer ${RINA_SRC} ${RINA_CUDA_SRC})
target_include_directories(rina_infer PUBLIC src)
target_link_libraries(rina_infer PRIVATE CUDA::cudart CUDA::cublas dl m)

set(RINA_TEST_CPP src/loader.cpp src/core/layer.cpp src/core/layers_fp32.cpp
    src/core/buffer.cpp src/core/tape.cpp)

macro(add_rina_test name src)
    add_executable(${name} ${RINA_TEST_CPP} ${RINA_CUDA_SRC} ${src})
    target_include_directories(${name} PUBLIC src)
    target_link_libraries(${name} PRIVATE CUDA::cudart CUDA::cublas dl m)
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

if(DEFINED TORCH_ROOT)
    target_link_libraries(rina_infer PRIVATE
        ${TORCH_ROOT}/lib/libtorch.so ${TORCH_ROOT}/lib/libc10.so
        ${TORCH_ROOT}/lib/libtorch_cpu.so ${TORCH_ROOT}/lib/libtorch_cuda.so)
    set_target_properties(rina_infer PROPERTIES
        BUILD_RPATH "${TORCH_ROOT}/lib" INSTALL_RPATH "${TORCH_ROOT}/lib")
endif()
