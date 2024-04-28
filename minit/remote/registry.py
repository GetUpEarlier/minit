from typing import Any

from types import FunctionType


FUNCTIONS = {}
OBJECT_TYPES = {}
VALUE_TYPES = {}
METHODS = {}
OBJECTS = {}


def register_function(fn: FunctionType):
    FUNCTIONS[fn.__qualname__] = fn
    return fn


def get_function(name: str) -> FunctionType:
    return FUNCTIONS[name]


def register_object(ty: type):
    OBJECT_TYPES[ty.__qualname__] = ty
    return ty


def is_object(ty: type):
    return ty.__qualname__ in OBJECT_TYPES


def create_object(obj: Any) -> int:
    id = len(OBJECTS)
    OBJECTS[id] = obj
    return id


def get_object(id: int):
    return OBJECTS[id]


def register_method(fn: FunctionType):
    return register_function(fn)


def register_value(ty):
    VALUE_TYPES[ty] = ()
    return ty


register_value(type(None))
register_value(bool)
register_value(int)
register_value(float)
register_value(str)
register_object(list)
register_method(list.append)
register_method(list.__getitem__)
register_method(list.__len__)
