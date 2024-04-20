from ctypes import CDLL
import functools
import os
from minit.compiler.cxx import CXXLibrary
from ...core.cache import cached
from ..toolkit import get_cuda_home


@cached()
def load_cuda_runtime():
    path = os.path.join(get_cuda_home(), "lib64", "libcudart.so")
    library = CDLL(path)
    return CXXLibrary(library)
