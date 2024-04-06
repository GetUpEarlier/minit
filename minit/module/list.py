from typing import Generic, Iterator, List, TypeVar
from .module import Module

_Module = TypeVar("_Module", bound=Module)

class ModuleList(Module, Generic[_Module]):
    _module_list: List[_Module]

    def __init__(self) -> None:
        super().__init__()
        self._module_list = []

    def append(self, module: _Module):
        index = len(self._module_list)
        self._module_list.append(self.register_module(str(index), module))

    def __getitem__(self, *args, **kwargs) -> _Module:
        return self._module_list.__getitem__(*args, **kwargs)

    def __iter__(self) -> Iterator[_Module]:
        return self._module_list.__iter__()

    def __len__(self) -> int:
        return self._module_list.__len__()
