from dataclasses import dataclass
from typing import Any

from ..core.scalar import ScalarTensor
from ..core.dispatch import register_dispatch
from ..core.operator import Operator


@dataclass
class Add(Operator):
    ...

@dataclass
class Subtract(Operator):
    ...

@dataclass
class Multiply(Operator):
    ...

@dataclass
class Divide(Operator):
    ...

@dataclass
class Power(Operator):
    ...

@dataclass
class Exponential(Operator):
    ...

@dataclass
class Cosine(Operator):
    ...

@dataclass
class Sine(Operator):
    ...

@dataclass
class Constant(Operator):
    value: Any
    dtype: str

@dataclass
class Cast(Operator):
    dtype: str

@register_dispatch()
def register_constant(op: Constant):
    c = ScalarTensor(op.value, op.dtype)
    return (c,)