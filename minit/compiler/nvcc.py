import ctypes
import os

from .cache import cached_execute
from .cxx import CXXCompiler, CXXLibrary, CXXUnit
from ..cuda.toolkit import get_cuda_home


class NVCC(CXXCompiler):
    def compile(self, unit: CXXUnit) -> CXXLibrary:
        commands = [os.path.join(get_cuda_home(), "bin", "nvcc")]
        for include in unit.includes:
            commands += ["-I", include]
        for library in unit.libraries:
            commands += ["-l", library]
        for define in unit.defines:
            commands += ["-D", define]
        commands += ["-shared"]
        commands += ["--compiler-options", "-fPIC"]
        commands += ["main.cu"]
        commands += ["-o", "library.so"]
        result = cached_execute(commands, {"main.cu": unit.source})
        library = ctypes.CDLL(os.path.join(result, "library.so"))
        return CXXLibrary(library=library, entrance=unit.entrance)


nvcc = NVCC()
