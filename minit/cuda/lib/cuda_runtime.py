from ctypes import CDLL
import functools
from minit.compiler.cxx import CXXLibrary


@functools.lru_cache(maxsize=None)
def load_cuda_runtime():
    path = "/usr/local/cuda/lib64/libcudart.so"
    library = CDLL(path)
    return CXXLibrary(library)
