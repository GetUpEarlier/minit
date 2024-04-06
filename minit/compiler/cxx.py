from ctypes import CDLL
from dataclasses import dataclass, field
from typing import Any, List, Optional


def find_library(name: str) -> str:
    ...

def find_include(name: str) -> str:
    ...


@dataclass
class CXXUnit:
    source: str
    entrance: Optional[str] = None
    includes: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    defines: List[str] = field(default_factory=list)


@dataclass
class CXXLibrary:
    library: CDLL
    entrance: Optional[str] = None

    def __call__(self, *args) -> Any:
        assert self.entrance is not None
        return getattr(self.library, self.entrance)(*args)


class CXXCompiler:
    def compile(self, unit: CXXUnit) -> CXXLibrary:
        ...
