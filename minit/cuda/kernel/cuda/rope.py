import ctypes
import functools

from ....core.cache import cached

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit, import_symbol
from ...compiler import nvcc

@cached()
def generate_rope_kernel(name: str, dtype: str):
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

__global__ void kernel(T* input, float* freqs_cos, float* freqs_sin, T* output, size_t batch_size, size_t seqlen, size_t nr_heads, size_t head_size) {
    size_t nr_complexes = batch_size * seqlen * nr_heads * (head_size/2);
    size_t stride = blockDim.x * gridDim.x;
    TensorIterator<4> input_iterator {batch_size, seqlen, nr_heads, head_size/2};
    TensorIterator<2> freqs_iterator {seqlen, head_size/2};
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_complexes; offset += stride) {
        auto input_index = input_iterator.to_index(offset);
        auto real = input[offset*2];
        auto imag = input[offset*2+1];
        auto freqs_offset = freqs_iterator.to_offset({input_index[1], input_index[3]});
        auto freq_cos = freqs_cos[freqs_offset];
        auto freq_sin = freqs_sin[freqs_offset];
        output[offset*2] = real * freq_cos - imag * freq_sin;
        output[offset*2+1] = imag * freq_cos + real * freq_sin;
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* input, float* freqs_cos, float* freqs_sin, T* output, size_t batch_size, size_t seqlen, size_t nr_heads, size_t head_size) {
    size_t nr_complexes = batch_size * seqlen * nr_heads * (head_size/2);
    if (nr_complexes == 0) {
        return;
    }
    static constexpr size_t nr_sms = 108;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_complexes + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_complexes + nr_threads_per_block - 1) / nr_threads_per_block;
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, freqs_cos, freqs_sin, output, batch_size, seqlen, nr_heads, head_size);
    CUDA_ASSERT(cudaGetLastError());
}
"""
    source = substitude(kernel_template, {
        "DATA_TYPE": dtype,
        "KERNEL_NAME": kernel_name,
    })
    kernel = nvcc.compile(CXXUnit(source=source))
    @import_symbol(kernel, kernel_name)
    def entrance(
        stream: ctypes.c_void_p,
        input: ctypes.c_void_p,
        freqs_cos: ctypes.c_void_p,
        freqs_sin: ctypes.c_void_p,
        output: ctypes.c_void_p,
        batch_size: ctypes.c_size_t,
        seqlen: ctypes.c_size_t,
        nr_heads: ctypes.c_size_t,
        head_size: ctypes.c_size_t,
    ):
        ...
    return entrance
