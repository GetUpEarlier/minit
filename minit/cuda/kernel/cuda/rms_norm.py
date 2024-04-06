import functools

from ....compiler.template import substitude
from ....compiler.cxx import CXXUnit
from ....compiler.nvcc import nvcc

@functools.lru_cache(maxsize=None)
def generate_rms_norm_kernel(name: str, dtype: str):
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
__launch_bounds__(nr_threads) __global__ void block_rms_norm_kernel(T* input, T* weight, T* output, size_t a, size_t b, size_t c, float eps) {
    typedef cub::BlockReduce<float, nr_threads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    size_t nr_lines = a * c;
    TensorIterator<2> line_iterator{a, c};
    TensorIterator<3> iterator{a, b, c};
    for (size_t line = blockIdx.x; line < nr_lines; line += gridDim.x) {
        auto line_index = line_iterator.to_index(line);
        __shared__ float sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            sum = 0;
        }
        size_t nr_loops = (b + blockDim.x - 1) / blockDim.x;
        for (size_t i = 0; i < nr_loops; ++i) {
            size_t index = i * blockDim.x + threadIdx.x;
            float value = index < b ? input[iterator.to_offset({line_index[0], index, line_index[1]})] : (T)0;
            value = value * value;
            float aggregate = BlockReduce(temp_storage).Reduce((float)value, cub::Sum(), nr_threads);
            if (threadIdx.x == 0) {
                sum = cub::Sum()(sum, aggregate);
            }
            __syncthreads();
        }
        float mean = sum / b;
        float rms = sqrt(mean + eps);
        for (size_t i = 0; i < nr_loops; ++i) {
            size_t index = i * blockDim.x + threadIdx.x;
            if (index < b) {
                float value = input[iterator.to_offset({line_index[0], index, line_index[1]})];
                value = value / rms;
                value *= (float)weight[index];
                output[iterator.to_offset({line_index[0], index, line_index[1]})] = value;
            }
        }
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* input, T* weight, T* output, size_t a, size_t b, size_t c, double eps) {
    static constexpr size_t nr_sms = 108;
    static constexpr size_t nr_threads_per_block = 1024;
    size_t nr_blocks = nr_sms;
    block_rms_norm_kernel<nr_threads_per_block><<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, weight, output, a, b, c, (float)eps);
}
"""
    source = substitude(kernel_template, {
        "DATA_TYPE": dtype,
        "KERNEL_NAME": kernel_name,
    })
    kernel = nvcc.compile(CXXUnit(entrance=kernel_name, source=source))
    return kernel
