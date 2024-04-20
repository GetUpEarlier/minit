import ctypes
import functools
import os

from ...core.cache import cached

from ...compiler.template import substitude
from ...compiler.cxx import CXXUnit, import_symbol
from ...compiler.gcc import gcc
from ..toolkit import find_cuda_include_directory, get_cuda_home

@cached()
def generate_cublas_kernel(name: str, dtype: str):
    kernel_name = f"minit_{name}"
    kernel_template =\
"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <unordered_map>
#include <stdexcept>


#define CUBLAS_ASSERT(expr)                                         \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != CUBLAS_STATUS_SUCCESS) {                        \\
            fprintf(stderr, "cublas errno: %d\\n", (int)_err);       \\
            throw std::runtime_error("cublas error at " #expr);     \\
        }                                                           \\
    } while (0)


#define CUDA_ASSERT(expr)                                           \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != cudaSuccess) {                                  \\
            throw std::runtime_error(cudaGetErrorString(_err));     \\
        }                                                           \\
    } while (0)

using T = ${DATA_TYPE};
static constexpr cudaDataType_t kCudaDataType = ${CUDA_DATA_TYPE};
static constexpr cublasComputeType_t kCudaComputeDataType = ${CUDA_COMPUTE_DATA_TYPE};

cublasLtMatrixLayout_t create_layout(size_t batch, size_t nr_rows, size_t nr_cols) {
    cublasLtMatrixLayout_t layout;
    CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout, kCudaDataType, nr_rows, nr_cols, nr_rows));
    int32_t batch_i32 = batch;
    int64_t batch_stride_i64 = nr_rows * nr_cols;
    CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_i32, sizeof(batch_i32)));
    CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_i64, sizeof(batch_stride_i64)));
    return layout;
}

cublasLtHandle_t get_handle() {
    static thread_local std::unordered_map<int, cublasLtHandle_t> handles;
    int device;
    CUDA_ASSERT(cudaGetDevice(&device));
    if (!handles.count(device)) {
        cublasLtHandle_t handle;
        CUBLAS_ASSERT(cublasLtCreate(&handle));
        handles[device] = handle;
    }
    return handles[device];
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, void* a, void* b, void* c, size_t batch, size_t m, size_t n, size_t k, void* workspace) {
    auto handle = get_handle();
    cublasLtMatmulDesc_t desc;
    CUBLAS_ASSERT(cublasLtMatmulDescCreate(&desc, kCudaComputeDataType, kCudaDataType));
    cublasLtMatmulPreference_t preference;
    CUBLAS_ASSERT(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspace_size = 0;
    CUBLAS_ASSERT(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
    cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N, transc = CUBLAS_OP_N;
    CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSC, &transc, sizeof(transc)));
    // CUBLAS_ASSERT(cublasLtMatmulPreferenceInit(preference));
    int heuristics_count = 1;
    cublasLtMatmulHeuristicResult_t heuristics[heuristics_count];
    // (k, m) @ (n, k) => (m, n)
    auto a_layout = create_layout(batch, k, n);
    auto b_layout = create_layout(batch, k, m);
    auto c_layout = create_layout(batch, n, m);
    CUBLAS_ASSERT(cublasLtMatmulAlgoGetHeuristic(handle, desc, a_layout, b_layout, c_layout, c_layout, preference, heuristics_count, heuristics, &heuristics_count));
    if (heuristics_count == 0) {
        fprintf(stderr, "no gemm algo found\\n");
    }
    float alpha = 1.0, beta = 1.0;
    CUDA_ASSERT(cudaMemset(c, 0, batch * m * n * sizeof(T)));
    CUBLAS_ASSERT(cublasLtMatmul(
        handle,
        desc,
        &alpha,
        b,
        a_layout,
        a,
        b_layout,
        &beta,
        c,
        c_layout,
        c,
        c_layout,
        &heuristics[0].algo,
        workspace,
        0,
        stream
    ));
}
"""
    source = substitude(kernel_template, {
        "KERNEL_NAME":  kernel_name,
        "DATA_TYPE": dtype,
        "CUDA_DATA_TYPE": {
            "float": "CUDA_R_32F",
            "__half": "CUDA_R_16F",
            "__nv_bfloat16": "CUDA_R_16BF",
        }[dtype],
        "CUDA_COMPUTE_DATA_TYPE": {
            "float": "CUBLAS_COMPUTE_32F",
            "__half": "CUBLAS_COMPUTE_32F",
            "__nv_bfloat16": "CUBLAS_COMPUTE_32F",
        }[dtype]
    })
    kernel = gcc.compile(CXXUnit(source=source, includes=[find_cuda_include_directory()], libraries=[
        os.path.join(get_cuda_home(), "lib64", "libcublasLt.so"),
        os.path.join(get_cuda_home(), "lib64", "libcudart.so"),
    ]))
    @import_symbol(kernel, kernel_name)
    def entrance(
        stream: ctypes.c_void_p,
        a: ctypes.c_void_p,
        b: ctypes.c_void_p,
        c: ctypes.c_void_p,
        batch: ctypes.c_size_t,
        m: ctypes.c_size_t,
        n: ctypes.c_size_t,
        k: ctypes.c_size_t,
        workspace: ctypes.c_void_p,
    ):
        ...
    return entrance
