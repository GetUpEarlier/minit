import ctypes
import functools

from .toolkit import find_cuda_include_directory, find_cuda_libraries
from ..compiler.cxx import CXXUnit
from ..compiler.gcc import gcc


@functools.lru_cache(maxsize=None)
def _generate_library():
    source =\
"""
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>


#define CUDA_ASSERT(expr)                                           \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != cudaSuccess) {                                  \\
            throw std::runtime_error(cudaGetErrorString(_err));     \\
        }                                                           \\
    } while (0)


extern "C" void* allocate_cuda_memory(size_t size) {
    void* pointer;
    CUDA_ASSERT(cudaMallocAsync(&pointer, size, nullptr));
    return pointer;
}

extern "C" void copy_cuda_memory(void* dst, const void* src, size_t size) {
    CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault));
}

extern "C" void free_cuda_memory(void* pointer) {
    CUDA_ASSERT(cudaFreeAsync(pointer, nullptr));
}

extern "C" void sync_cuda() {
    CUDA_ASSERT(cudaStreamSynchronize(nullptr));
}
"""
    return gcc.compile(CXXUnit(source=source, libraries=find_cuda_libraries(), includes=[
        find_cuda_include_directory()
    ]))


def allocate_cuda_memory(size: int):
    _generate_library().library.allocate_cuda_memory.restype = ctypes.c_void_p
    return _generate_library().library.allocate_cuda_memory(ctypes.c_size_t(size))


def free_cuda_memory(pointer: int):
    _generate_library().library.free_cuda_memory(ctypes.c_void_p(pointer))


def copy_cuda_memory(dst: int, src: int, size: int):
    _generate_library().library.copy_cuda_memory(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(size))


def sync_cuda():
    _generate_library().library.sync_cuda()


class CUDAMemory:
    _pointer = None
    _size = None

    def __init__(self, size: int) -> None:
        self._pointer = allocate_cuda_memory(size)
        self._size = size


    def __del__(self):
        free_cuda_memory(self._pointer)
        self._pointer = None
        self._size = None


    def copy_from(self, src: "CUDAMemory"):
        assert self._size == src._size
        copy_cuda_memory(self._pointer, src._pointer, self._size)


    def copy(self) -> "CUDAMemory":
        new = CUDAMemory(self._size)
        copy_cuda_memory(new._pointer, self._pointer, self._size)
        return new
