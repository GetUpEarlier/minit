def find_cuda_library_directory():
    return "/usr/local/cuda/lib64/"


def find_cuda_include_directory():
    return "/usr/local/cuda/include/"


def find_cuda_libraries():
    libraries = [
        find_cuda_library_directory() + "libcudart.so"
    ]
    return libraries
