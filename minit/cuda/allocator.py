import ctypes
import functools
import nvtx

from ..core.cache import cached

from .toolkit import find_cuda_include_directory, find_cuda_libraries
from ..compiler.cxx import CXXLibrary, CXXUnit, import_symbol
from ..compiler.gcc import gcc


@cached()
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
    CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, nullptr));
}

extern "C" void free_cuda_memory(void* pointer) {
    CUDA_ASSERT(cudaFreeAsync(pointer, nullptr));
}

extern "C" void sync_cuda() {
    CUDA_ASSERT(cudaStreamSynchronize(nullptr));
}

extern "C" void reset(void* pointer, size_t size) {
    CUDA_ASSERT(cudaMemsetAsync(pointer, 0, size, nullptr));
}
"""
    return gcc.compile(CXXUnit(source=source, libraries=find_cuda_libraries(), includes=[
        find_cuda_include_directory()
    ]))


_library = _generate_library()


@import_symbol(_library, "allocate_cuda_memory")
def allocate_cuda_memory(size: ctypes.c_size_t) -> ctypes.c_void_p:
    ...


@import_symbol(_library, "free_cuda_memory")
def free_cuda_memory(pointer: ctypes.c_void_p):
    ...


@import_symbol(_library, "copy_cuda_memory")
def copy_cuda_memory(dst: ctypes.c_void_p, src: ctypes.c_void_p, size: ctypes.c_size_t):
    ...


@import_symbol(_library, "sync_cuda")
def sync_cuda():
    ...


@import_symbol(_library, "reset")
def reset(dst: ctypes.c_void_p, size: ctypes.c_size_t):
    ...


class CUDAMemory:
    __slots__ = [
        "_pointer",
        "_size",
    ]

    _pointer: int
    _size: int

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


    def reset(self):
        reset(self._pointer, self._size)
