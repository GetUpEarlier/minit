import ctypes
import functools

from ....core.cache import cached

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit, import_symbol
from ...compiler import nvcc

@cached()
def generate_elemwise_kernel(name: str, nr_inputs: int, expr: str, dtype: str):
    kernel_name = f"minit_elemwise_{name}"
    kernel_template =\
"""
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda/std/array>
#include <cstring>
#include <cstdio>
#include <stdexcept>


#define CUDA_ASSERT(expr)                                           \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != cudaSuccess) {                                  \\
            throw std::runtime_error(cudaGetErrorString(_err));     \\
        }                                                           \\
    } while (0)

using T = ${DATA_TYPE};
static constexpr size_t nr_inputs = ${NR_INPUTS};

__global__ void kernel(cuda::std::array<T*, nr_inputs> inputs, cuda::std::array<int, nr_inputs> strides, T* output, size_t nr_elements) {
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_elements; offset += blockDim.x * gridDim.x) {
        T values[nr_inputs];
#pragma unroll
        for (size_t i = 0; i < nr_inputs; ++i) {
            values[i] = __ldg(inputs[i] + offset*strides[i]);
        }
        output[offset] = ${EXPR};
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T** inputs, int* strides, T* output, size_t nr_elements) {
    if (nr_elements == 0) {
        return;
    }
    static constexpr size_t nr_sms = 108;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_elements + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_elements + nr_threads_per_block - 1) / nr_threads_per_block;
    cuda::std::array<T*, nr_inputs> inputs_array;
    std::memcpy(&inputs_array, inputs, sizeof(inputs_array));
    cuda::std::array<int, nr_inputs> stride_array;
    std::memcpy(&stride_array, strides, sizeof(stride_array));
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(inputs_array, stride_array, output, nr_elements);
    CUDA_ASSERT(cudaGetLastError());
}
"""
    source = substitude(kernel_template, {
        "DATA_TYPE": dtype,
        "NR_INPUTS": str(nr_inputs),
        "EXPR": expr.format(*[f"values[{i}]" for i in range(nr_inputs)]),
        "KERNEL_NAME": kernel_name,
    })
    kernel = nvcc.compile(CXXUnit(source=source))
    @import_symbol(kernel, kernel_name)
    def entrance(
        stream: ctypes.c_void_p,
        input: ctypes.c_void_p,
        strides: ctypes.c_void_p,
        output: ctypes.c_void_p,
        nr_elements: ctypes.c_size_t,
    ):
        ...
    return entrance
