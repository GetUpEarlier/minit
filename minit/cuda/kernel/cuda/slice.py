import ctypes
import functools

from ....core.cache import cached

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit, import_symbol
from ...compiler import nvcc

@cached()
def generate_slice_kernel(name: str, dtype: str):
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

__global__ void kernel(T* input, T* output, size_t a, size_t b, size_t c, size_t start, size_t stop) {
    size_t nr_elements = a * c * (stop - start);
    size_t stride = blockDim.x * gridDim.x;
    TensorIterator<3> input_iterator;
    input_iterator.shape[0] = a;
    input_iterator.shape[1] = b;
    input_iterator.shape[2] = c;
    TensorIterator<3> output_iterator;
    output_iterator.shape[0] = a;
    output_iterator.shape[1] = stop - start;
    output_iterator.shape[2] = c;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_elements; offset += stride) {
        auto index = output_iterator.to_index(offset);
        index[1] += start;
        output[offset] = input[input_iterator.to_offset(index)];
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* input, T* output, size_t a, size_t b, size_t c, size_t start, size_t stop) {
    size_t nr_elements = a * c * (stop - start);
    if (nr_elements == 0) {
        return;
    }
    static constexpr size_t nr_sms = 108;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_elements + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_elements + nr_threads_per_block - 1) / nr_threads_per_block;
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, output, a, b, c, start, stop);
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
        output: ctypes.c_void_p,
        a: ctypes.c_size_t,
        b: ctypes.c_size_t,
        c: ctypes.c_size_t,
        start: ctypes.c_size_t,
        stop: ctypes.c_size_t,
    ):
        ...
    return entrance


@cached()
def generate_slice_set_kernel(name: str, dtype: str):
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

__global__ void kernel(T* input, T* output, size_t a, size_t b, size_t c, size_t start, size_t stop) {
    size_t nr_elements = a * c * (stop - start);
    size_t stride = blockDim.x * gridDim.x;
    TensorIterator<3> input_iterator;
    input_iterator.shape[0] = a;
    input_iterator.shape[1] = stop - start;
    input_iterator.shape[2] = c;
    TensorIterator<3> output_iterator;
    output_iterator.shape[0] = a;
    output_iterator.shape[1] = b;
    output_iterator.shape[2] = c;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_elements; offset += stride) {
        auto index = input_iterator.to_index(offset);
        index[1] += start;
        output[output_iterator.to_offset(index)] = input[offset];
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* input, T* output, size_t a, size_t b, size_t c, size_t start, size_t stop) {
    size_t nr_elements = a * c * (stop - start);
    if (nr_elements == 0) {
        return;
    }
    static constexpr size_t nr_sms = 108;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_elements + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_elements + nr_threads_per_block - 1) / nr_threads_per_block;
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, output, a, b, c, start, stop);
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
        output: ctypes.c_void_p,
        a: ctypes.c_size_t,
        b: ctypes.c_size_t,
        c: ctypes.c_size_t,
        start: ctypes.c_size_t,
        stop: ctypes.c_size_t,
    ):
        ...
    return entrance
