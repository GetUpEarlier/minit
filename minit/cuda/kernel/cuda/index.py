import functools

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit
from ....compiler.nvcc import nvcc

@functools.lru_cache(maxsize=None)
def generate_index_kernel(name: str, dtype: str):
    kernel_name = f"minit_{name}"
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

template <size_t nr_ranks>
struct TensorIterator {
    size_t shape[nr_ranks];

    __device__ cuda::std::array<size_t, nr_ranks> to_index(size_t offset) const {
        cuda::std::array<size_t, nr_ranks> index;
        for (size_t i = 0; i < nr_ranks; ++i) {
            index[nr_ranks-i-1] = offset % shape[nr_ranks-i-1];
            offset /= shape[nr_ranks-i-1];
        }
        return index;
    }

    __device__ size_t to_offset(cuda::std::array<size_t, nr_ranks> index) const {
        size_t offset = 0;
        for (size_t i = 0; i < nr_ranks; ++i) {
            offset *= shape[i];
            offset += index[i];
        }
        return offset;
    }
};

__global__ void kernel(T* input, std::int32_t* index, T* output, size_t a, size_t b, size_t c, size_t d) {
    size_t nr_elements = a * c * d;
    size_t stride = blockDim.x * gridDim.x;
    TensorIterator<3> output_iterator;
    output_iterator.shape[0] = a;
    output_iterator.shape[1] = c;
    output_iterator.shape[2] = d;
    TensorIterator<3> input_iterator;
    input_iterator.shape[0] = a;
    input_iterator.shape[1] = b;
    input_iterator.shape[2] = d;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_elements; offset += stride) {
        auto output_offset = offset;
        auto output_index = output_iterator.to_index(output_offset);
        if (index[output_index[1]] < 0) {
            __trap();
        }
        if (index[output_index[1]] >= b) {
            __trap();
        }
        auto input_offset = input_iterator.to_offset({output_index[0], index[output_index[1]], output_index[2]});
        output[offset] = input[input_offset];
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* input, std::int32_t* index, T* output, size_t a, size_t b, size_t c, size_t d) {
    size_t nr_elements = a * c * d;
    if (nr_elements == 0) {
        return;
    }
    static constexpr size_t nr_sms = 108;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_elements + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_elements + nr_threads_per_block - 1) / nr_threads_per_block;
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, index, output, a, b, c, d);
    CUDA_ASSERT(cudaGetLastError());
}
"""
    source = substitude(kernel_template, {
        "DATA_TYPE": dtype,
        "KERNEL_NAME": kernel_name,
    })
    kernel = nvcc.compile(CXXUnit(entrance=kernel_name, source=source))
    return kernel
