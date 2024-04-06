from typing import Dict, Generic, Tuple, TypeVar

from ..core.tensor import Tensor

_Child = TypeVar("_Child")
_Parent = TypeVar("_Parent")

class Module(Generic[_Parent]):
    parent: _Parent
    children: Dict[str, "Module"]
    buffers: Dict[str, Tensor]

    def __init__(self) -> None:
        super().__init__()
        self.children = {}
        self.buffers = {}

    def register_module(self, name: str, module: _Child) -> _Child:
        self.children[name] = module
        return module

    def register_buffer(self, name: str, shape: Tuple[int, ...], dtype: str):
        from ..cuda.tensor import CUDATensor
        buffer = CUDATensor.allocate(shape, dtype)
        self.buffers[name] = buffer
        return buffer

    def named_buffers(self):
        for name, buffer in self.buffers.items():
            yield name, buffer
        for module_name, child in self.children.items():
            for name, buffer in child.named_buffers():
                yield module_name + "." + name, buffer
