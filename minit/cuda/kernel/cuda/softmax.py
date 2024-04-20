import ctypes
import functools

from ....core.cache import cached

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit, import_symbol
from ...compiler import nvcc

@cached()
def generate_softmax_kernel(name: str, dtype: str):
    kernel_name = f"minit_{name}"
    kernel_template =\
"""
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda/std/array>
#include <cub/cub.cuh>
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

template <size_t nr_threads>
__launch_bounds__(nr_threads) __global__ void softmax_kernel(T* input, T* output, size_t a, size_t b, size_t c) {
    typedef cub::BlockReduce<float, nr_threads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    size_t nr_lines = a * c;
    TensorIterator<3> iterator{a, b, c};
    TensorIterator<2> line_iterator{a, c};
    for (size_t line = blockIdx.x; line < nr_lines; line += gridDim.x) {
        auto line_index = line_iterator.to_index(line);
        __shared__ float max;
        __shared__ float sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            max = -INFINITY;
            sum = 0;
        }
        size_t nr_loops = (b + blockDim.x - 1) / blockDim.x;
        for (size_t i = 0; i < nr_loops; ++i) {
            size_t index = i * blockDim.x + threadIdx.x;
            float value = index < b ? (float)input[iterator.to_offset({line_index[0], index, line_index[1]})] : -INFINITY;
            float aggregate = BlockReduce(temp_storage).Reduce((float)value, cub::Max(), nr_threads);
            if (threadIdx.x == 0) {
                max = cub::Max()(max, aggregate);
            }
            __syncthreads();
        }
        for (size_t i = 0; i < nr_loops; ++i) {
            size_t index = i * blockDim.x + threadIdx.x;
            float value = 0;
            if (index < b) {
                value = (float)input[iterator.to_offset({line_index[0], index, line_index[1]})];
                value -= max;
                value = expf(value);
            }
            float aggregate = BlockReduce(temp_storage).Reduce((float)value, cub::Sum(), nr_threads);
            if (threadIdx.x == 0) {
                sum = cub::Sum()(sum, aggregate);
            }
            __syncthreads();
        }
        for (size_t i = 0; i < nr_loops; ++i) {
            size_t index = i * blockDim.x + threadIdx.x;
            if (index < b) {
                float value = input[iterator.to_offset({line_index[0], index, line_index[1]})];
                value -= max;
                value = expf(value);
                value = value / sum;
                output[iterator.to_offset({line_index[0], index, line_index[1]})] = (T)value;
            }
            __syncthreads();
        }
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* input, T* output, size_t a, size_t b, size_t c) {
    static constexpr size_t nr_sms = 108;
    static constexpr size_t nr_threads_per_block = 1024;
    size_t nr_blocks = nr_sms;
    softmax_kernel<nr_threads_per_block><<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, output, a, b, c);
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
    ):
        ...
    return entrance
