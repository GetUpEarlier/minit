import functools

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit
from ....compiler.nvcc import nvcc

@functools.lru_cache(maxsize=None)
def generate_elemwise_kernel(name: str, nr_inputs: int, expr: str):
    kernel_name = f"minit_elemwise_{name}"
    kernel_template =\
"""
#include <cuda.h>
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

__global__ void kernel(cuda::std::array<T*, nr_inputs> inputs, T* output, size_t nr_elements) {
    size_t stride = blockDim.x * gridDim.x;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_elements; offset += stride) {
        T values[nr_inputs];
#pragma unroll
        for (size_t i = 0; i < nr_inputs; ++i) {
            values[i] = inputs[i][offset];
        }
        output[offset] = ${EXPR};
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T** inputs, T* output, size_t nr_elements) {
    if (nr_elements == 0) {
        return;
    }
    static constexpr size_t nr_sms = 108;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_elements + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_elements + nr_threads_per_block - 1) / nr_threads_per_block;
    cuda::std::array<T*, nr_inputs> inputs_array;
    std::memcpy(&inputs_array, inputs, sizeof(inputs_array));
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(inputs_array, output, nr_elements);
    CUDA_ASSERT(cudaGetLastError());
}
"""
    source = substitude(kernel_template, {
        "DATA_TYPE": "float",
        "NR_INPUTS": str(nr_inputs),
        "EXPR": expr,
        "KERNEL_NAME": kernel_name,
    })
    kernel = nvcc.compile(CXXUnit(entrance=kernel_name, source=source))
    return kernel
