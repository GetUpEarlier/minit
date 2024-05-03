import os
from types import CodeType

from .cache import cached_execute


class PythonCompiler:
    def compile(self, source: str) -> CodeType:
        path = cached_execute([], {
            "kernel.py": source
        })
        kernel_code = compile(source, os.path.join(path, "kernel.py"), "exec")
        globals = {}
        exec(kernel_code, globals)
        return globals


pythonc = PythonCompiler()
