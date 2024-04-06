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

    cuda::std::array<size_t, nr_ranks> to_index(size_t offset) const {
        cuda::std::array<size_t, nr_ranks> index;
        for (size_t i = 0; i < nr_ranks; ++i) {
            index[nr_ranks-i-1] = offset % shape[nr_ranks-i-1];
            offset /= shape[nr_ranks-i-1];
        }
        return index;
    }

    size_t to_offset(cuda::std::array<size_t, nr_ranks> index) const {
        size_t offset = 0;
        for (size_t i = 0; i < nr_ranks; ++i) {
            offset *= shape[i];
            offset += index[i];
        }
        return offset;
    }
};

// input[a, b, c, d, e] -> output[a, d, c, b, e]
__global__ void kernel(T* input, T* output, size_t a, size_t b, size_t c) {
    size_t stride = blockDim.x * gridDim.x;
    size_t nr_lines = a * c;
    // TensorIterator<3> input_iterator{a, b, c};
    // TensorIterator<2> output_iterator{a, c};
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < nr_lines; offset += stride) {
        T result = ${REDUCE_INIT};
        for (size_t j = 0; j < b; j++) {
            T x = result;
            T y = input[(offset / c) * (b*c) + offset%c + j * c];
            result = ${REDUCE_EXPR};
        }
        output[offset] = result;
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* input, T* output, size_t a, size_t b, size_t c) {
    static constexpr size_t nr_sms = 112;
    size_t nr_lines = a * c;
    size_t nr_threads_per_block = std::min((size_t)1024, (size_t)((nr_lines + nr_sms - 1) / nr_sms));
    size_t nr_blocks = (nr_lines + nr_threads_per_block - 1) / nr_threads_per_block;
    kernel<<<nr_blocks, nr_threads_per_block, 0, stream>>>(input, output, a, b, c);
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
