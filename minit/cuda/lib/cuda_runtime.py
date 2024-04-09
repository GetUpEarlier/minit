from ctypes import CDLL
import functools
import os
from minit.compiler.cxx import CXXLibrary
from ..toolkit import get_cuda_home


@functools.lru_cache(maxsize=None)
def load_cuda_runtime():
    path = os.path.join(get_cuda_home(), "lib64", "libcudart.so")
    library = CDLL(path)
    return CXXLibrary(library)
