import os


_cuda_home = None


def get_cuda_home():
    global _cuda_home
    if _cuda_home is None:
        _cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
    return _cuda_home


def find_cuda_library_directory():
    return os.path.join(get_cuda_home(), "lib64")


def find_cuda_include_directory():
    return os.path.join(get_cuda_home(), "include")


def find_cuda_libraries():
    libraries = [
        os.path.join(find_cuda_library_directory(), "libcudart.so"),
    ]
    return libraries
