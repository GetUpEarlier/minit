import functools

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit
from ....compiler.nvcc import nvcc

@functools.lru_cache(maxsize=None)
def generate_fill_kernel(name: str):
    kernel_name = f"minit_fill_{name}"
    kernel_template =\
"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>


#define CUDA_ASSERT(expr)                                           \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != cudaSuccess) {                                  \\
            throw std::runtime_error(cudaGetErrorString(_err));     \\
        }                                                           \\
    } while (0)

using T = ${DATA_TYPE};

__global__ void kernel(T* output, size_t nr_elements, T value) {
    size_t stride = blockDim.x * gridDim.x;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_elements; offset += stride) {
        output[offset] = value;
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* output, size_t nr_elements, T value) {
    if (nr_elements == 0) {
        return;
    }
    static constexpr size_t nr_sms = 112;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_elements + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_elements + nr_threads_per_block - 1) / nr_threads_per_block;
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(output, nr_elements, value);
}
"""
    source = substitude(kernel_template, {
        "DATA_TYPE": "float",
        "KERNEL_NAME": kernel_name,
    })
    kernel = nvcc.compile(CXXUnit(entrance=kernel_name, source=source))
    return kernel
