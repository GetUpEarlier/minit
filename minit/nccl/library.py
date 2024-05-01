import ctypes
import functools
import os
import nvtx
import nvidia.nccl

from ..compiler.template import substitude
from ..core.cache import cached
from ..cuda.toolkit import find_cuda_include_directory, find_cuda_libraries
from ..compiler.cxx import CXXLibrary, CXXUnit, import_symbol
from ..compiler.gcc import gcc


@cached()
def _generate_nccl_library():
    kernel_template =\
"""
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <nccl.h>


#define CUDA_ASSERT(expr)                                           \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != cudaSuccess) {                                  \\
            throw std::runtime_error(cudaGetErrorString(_err));     \\
        }                                                           \\
    } while (0)


#define NCCL_ASSERT(expr)                                           \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != ncclSuccess) {                                  \\
            throw std::runtime_error(ncclGetErrorString(_err));     \\
        }                                                           \\
    } while (0)

extern "C" size_t nccl_unique_id_size() {
    return sizeof(ncclUniqueId);
}

extern "C" void nccl_create_unique_id(char* bytes) {
    ncclUniqueId unique_id;
    NCCL_ASSERT(ncclGetUniqueId(&unique_id));
    std::memcpy(bytes, &unique_id, sizeof(unique_id));
}

extern "C" ncclComm_t nccl_init_rank(int nr_ranks, int rank, char* bytes) {
    ncclUniqueId unique_id;
    std::memcpy(&unique_id, bytes, sizeof(unique_id));
    CUDA_ASSERT(cudaSetDevice(rank));
    ncclComm_t comm;
    NCCL_ASSERT(ncclCommInitRank(&comm, nr_ranks, unique_id, rank));
    return comm;
}
"""
    kernel_source = substitude(kernel_template, {})
    library = gcc.compile(CXXUnit(
        source=kernel_source,
        libraries=[
            *find_cuda_libraries(),
            os.path.join(nvidia.nccl.__path__[0], "lib", "libnccl.so.2"),
        ], includes=[
            find_cuda_include_directory(),
            os.path.join(nvidia.nccl.__path__[0], "include"),
        ]
    ))
    return library


_library = _generate_nccl_library()

@import_symbol(_library, "nccl_unique_id_size")
def nccl_unique_id_size() -> ctypes.c_size_t:
    ...

@import_symbol(_library, "nccl_create_unique_id")
def nccl_create_unique_id(
    bytes: ctypes.c_void_p,
):
    ...

@import_symbol(_library, "nccl_init_rank")
def nccl_init_rank(
    nr_ranks: ctypes.c_int,
    rank: ctypes.c_int,
    bytes: ctypes.c_void_p,
) -> ctypes.c_void_p:
    ...

def launch_server() -> bytes:
    size = nccl_unique_id_size()
    id = bytearray(size)
    nccl_create_unique_id((ctypes.c_char * size).from_buffer(id))
    return bytes(id)

def connect_server(nr_ranks: int, rank: int, id: bytes) -> int:
    return nccl_init_rank(nr_ranks, rank, id)
