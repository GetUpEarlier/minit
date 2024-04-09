import functools

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit
from ....compiler.nvcc import nvcc

@functools.lru_cache(maxsize=None)
def generate_cast_kernel(name: str, source_dtype: str, target_dtype: str):
    kernel_name = f"minit_cast_{name}"
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

using source_type = ${SOURCE_TYPE};
using target_type = ${TARGET_TYPE};

__global__ void kernel(source_type* input, target_type* output, size_t nr_elements) {
    size_t stride = blockDim.x * gridDim.x;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_elements; offset += stride) {
        auto value = input[offset];
        output[offset] = (target_type)value;
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, source_type* input, target_type* output, size_t nr_elements) {
    if (nr_elements == 0) {
        return;
    }
    static constexpr size_t nr_sms = 108;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_elements + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_elements + nr_threads_per_block - 1) / nr_threads_per_block;
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, output, nr_elements);
    CUDA_ASSERT(cudaGetLastError());
}
"""
    source = substitude(kernel_template, {
        "SOURCE_TYPE": source_dtype,
        "TARGET_TYPE": target_dtype,
        "KERNEL_NAME": kernel_name,
    })
    kernel = nvcc.compile(CXXUnit(entrance=kernel_name, source=source))
    return kernel
