from ctypes import CDLL
import ctypes
from dataclasses import dataclass, field
import inspect
from types import FunctionType
from typing import Any, Dict, List, Optional
import nvtx


def find_library(name: str) -> str:
    ...

def find_include(name: str) -> str:
    ...


@dataclass
class CXXUnit:
    source: str
    includes: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    defines: List[str] = field(default_factory=list)


class CXXLibrary:
    __slots__ = [
        "library",
        "symbols",
    ]

    library: CDLL
    symbols: Dict[str, ctypes._CFuncPtr]

    def __init__(self, library: CDLL):
        self.library = library
        self.symbols = {}


def import_symbol(cdll: CDLL, name: str):
    def decorator(function: FunctionType):
        arg_types = []
        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        for param in parameters:
            assert param.kind in [param.POSITIONAL_ONLY , param.POSITIONAL_OR_KEYWORD]
            arg_types.append(param.annotation)
        if signature.return_annotation is signature.empty:
            res_type = None
        else:
            res_type = signature.return_annotation
        symbol = getattr(cdll, name)
        symbol.restype = res_type
        symbol.argtypes = arg_types
        # @nvtx.annotate(name)
        # def decorated(*args):
        #     return symbol(*args)
        # return decorated
        return symbol
    return decorator


class CXXCompiler:
    def compile(self, unit: CXXUnit) -> CDLL:
        ...
