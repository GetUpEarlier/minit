import functools

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit
from ....compiler.nvcc import nvcc

@functools.lru_cache(maxsize=None)
def generate_reduce_kernel(name: str, init: str, expr: str):
    kernel_name = f"minit_{name}"
    kernel_template =\
"""
#include <cuda.h>
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

__global__ void thread_reduce_kernel(T* input, T* output, size_t a, size_t b, size_t c) {
    size_t stride = blockDim.x * gridDim.x;
    size_t nr_lines = a * c;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_lines; offset += stride) {
        T result = ${REDUCE_INIT};
        for (size_t j = 0; j < b; j++) {
            T x = result;
            T y = input[(offset / c) * (b*c) + offset%c + j * c];
            result = ${REDUCE_EXPR}(x, y);
        }
        output[offset] = result;
    }
}

template <size_t nr_threads>
__launch_bounds__(nr_threads) __global__ void block_reduce_kernel(T* input, T* output, size_t a, size_t b, size_t c) {
    typedef cub::BlockReduce<T, nr_threads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    size_t nr_lines = a * c;
    TensorIterator<3> input_iterator{a, b, c};
    TensorIterator<2> output_iterator{a, c};
    for (size_t line = blockIdx.x; line < nr_lines; line += gridDim.x) {
        auto output_index = output_iterator.to_index(line);
        T result = ${REDUCE_INIT};
        size_t nr_loops = (b + blockDim.x - 1) / blockDim.x;
        for (size_t i = 0; i < nr_loops; ++i) {
            size_t index = i * blockDim.x + threadIdx.x;
            T value = index < b ? input[input_iterator.to_offset({output_index[0], index, output_index[1]})] : ${REDUCE_INIT};
            T aggregate = BlockReduce(temp_storage).Reduce(value, ${REDUCE_EXPR}, nr_threads);
            if (threadIdx.x == 0) {
                result = ${REDUCE_EXPR}(result, aggregate);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            output[line] = result;
        }
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* input, T* output, size_t a, size_t b, size_t c) {
    static constexpr size_t nr_sms = 108;
    static constexpr size_t nr_threads_per_block = 1024;
    size_t nr_blocks = nr_sms;
    block_reduce_kernel<nr_threads_per_block><<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, output, a, b, c);
}
"""
    source = substitude(kernel_template, {
        "DATA_TYPE": "float",
        "KERNEL_NAME": kernel_name,
        "REDUCE_INIT": init,
        "REDUCE_EXPR": expr,
    })
    kernel = nvcc.compile(CXXUnit(entrance=kernel_name, source=source))
    return kernel
