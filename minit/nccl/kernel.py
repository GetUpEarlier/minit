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


NCCL_DTYPE_MAPPING = {
    "bfloat16": "ncclBfloat16",
    "float16": "ncclFloat16",
    "float32": "ncclFloat32",
    "float64": "ncclFloat64",
    "int8": "ncclInt8",
    "int32": "ncclInt32",
    "int64": "ncclInt64",
    "uint8": "ncclUint8",
    "uint32": "ncclUint32",
    "uint64": "ncclUint64",
}


@cached()
def _generate_nccl_primitives(dtype: str):
    kernel_template =\
"""
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <nccl.h>


#define NCCL_ASSERT(expr)                                           \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != ncclSuccess) {                                  \\
            throw std::runtime_error(ncclGetErrorString(_err));     \\
        }                                                           \\
    } while (0)


extern "C" void nccl_all_gather(cudaStream_t stream, ncclComm_t comm, const void* sendbuff, void* recvbuff, size_t size) {
    NCCL_ASSERT(ncclAllGather(sendbuff, recvbuff, size, ${NCCL_DATA_TYPE}, comm, stream));
}

extern "C" void nccl_all_reduce(cudaStream_t stream, ncclComm_t comm, const void* sendbuff, void* recvbuff, size_t size) {
    NCCL_ASSERT(ncclAllReduce(sendbuff, recvbuff, size, ${NCCL_DATA_TYPE}, ncclSum, comm, stream));
}

extern "C" void nccl_broadcast(cudaStream_t stream, ncclComm_t comm, const void* sendbuff, void* recvbuff, size_t size, int root) {
    NCCL_ASSERT(ncclBroadcast(sendbuff, recvbuff, size, ${NCCL_DATA_TYPE}, root, comm, stream));
}
"""
    kernel_source = substitude(kernel_template, {
        "NCCL_DATA_TYPE": NCCL_DTYPE_MAPPING[dtype],
    })
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
    @import_symbol(library, "nccl_all_gather")
    def nccl_all_gather(
        stream: ctypes.c_void_p,
        comm: ctypes.c_void_p,
        sendbuff: ctypes.c_void_p,
        recvbuff: ctypes.c_void_p,
        size: ctypes.c_size_t,
    ):
        ...
    @import_symbol(library, "nccl_all_reduce")
    def nccl_all_reduce(
        stream: ctypes.c_void_p,
        comm: ctypes.c_void_p,
        sendbuff: ctypes.c_void_p,
        recvbuff: ctypes.c_void_p,
        size: ctypes.c_size_t,
    ):
        ...
    @import_symbol(library, "nccl_broadcast")
    def nccl_broadcast(
        stream: ctypes.c_void_p,
        comm: ctypes.c_void_p,
        sendbuff: ctypes.c_void_p,
        recvbuff: ctypes.c_void_p,
        size: ctypes.c_size_t,
        root: ctypes.c_int,
    ):
        ...
    return nccl_all_gather, nccl_all_reduce, nccl_broadcast
