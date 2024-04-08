from typing import Any, Dict, Generic, Optional, Set, Tuple, TypeVar

from ..core.tensor import Tensor

_Child = TypeVar("_Child")
_Parent = TypeVar("_Parent")

class Module(Generic[_Parent]):
    parent: _Parent
    children: Dict[str, Any]
    buffers: Dict[str, Tuple[()]]

    def __init__(self) -> None:
        super().__init__()
        self.children = {}
        self.buffers = {}

    def register_module(self, name: str, module: _Child) -> _Child:
        self.children[name] = ()
        return module

    def register_buffer(self, name: str, shape: Tuple[int, ...], dtype: str, buffer: Optional[Tensor] = None):
        from ..cuda.tensor import CUDATensor
        if buffer is None:
            buffer = CUDATensor.allocate(shape, dtype)
        self.buffers[name] = ()
        return buffer

    def named_buffers(self):
        for name, buffer in self.buffers.items():
            yield name, buffer
        for module_name, child in self.children.items():
            for name, buffer in child.named_buffers():
                yield module_name + "." + name, buffer

    def get_child(self, name: str) -> "Module":
        if "." in name:
            prefix, suffix = name.split(".", maxsplit=1)
            return self.get_child(prefix).get_child(suffix)
        else:
            assert name in self.children
            if name[0].isdigit():
                return self[int(name)]
            else:
                return getattr(self, name)

    def get_buffer(self, name: str) -> Tensor:
        if "." in name:
            prefix, suffix = name.split(".", maxsplit=1)
            return self.get_child(prefix).get_buffer(suffix)
        else:
            assert name in self.buffers
            if name[0].isdigit():
                return self[int(name)]
            else:
                return getattr(self, name)

    def update_buffer(self, name: str, value: Tensor) -> Tensor:
        if "." in name:
            prefix, suffix = name.split(".", maxsplit=1)
            self.get_child(prefix).update_buffer(suffix, value)
        else:
            assert name in self.buffers
            if name[0].isdigit():
                self[int(name)] = value
            else:
                setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
