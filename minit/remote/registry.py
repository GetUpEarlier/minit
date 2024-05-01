from typing import Any, Tuple

from types import FunctionType


FUNCTIONS = {}
CONSTRUCTORS = {}
OBJECTS = {}


def register_function(fn: FunctionType):
    FUNCTIONS[fn.__qualname__] = fn
    return fn


def register_constructor(fn: FunctionType):
    FUNCTIONS[fn.__qualname__] = fn
    CONSTRUCTORS[fn.__qualname__] = fn
    return fn


def get_function(name: str) -> Tuple[FunctionType, bool]:
    return (FUNCTIONS[name],name in CONSTRUCTORS)


def create_object(obj: Any) -> int:
    id = len(OBJECTS)
    OBJECTS[id] = obj
    return id


def get_object(id: int):
    return OBJECTS[id]


def register_method(fn: FunctionType):
    return register_function(fn)


register_method(list.append)
register_method(list.__getitem__)
register_method(list.__len__)
