import ctypes
import os

from .cache import cached_execute
from .cxx import CXXCompiler, CXXLibrary, CXXUnit


class GCC(CXXCompiler):
    def compile(self, unit: CXXUnit) -> CXXLibrary:
        commands = ["g++"]
        for include in unit.includes:
            commands += ["-I", include]
        for define in unit.defines:
            commands += ["-D", define]
        commands += ["-shared"]
        commands += ["-fPIC"]
        commands += ["-O0"]
        commands += ["-g"]
        commands += ["main.cpp"]
        commands += ["-o", "library.so"]
        for library in unit.libraries:
            commands += [library]
        result = cached_execute(commands, {"main.cpp": unit.source})
        library = ctypes.CDLL(os.path.join(result, "library.so"))
        return library


gcc = GCC()
