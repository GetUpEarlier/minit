from dataclasses import dataclass
from typing import Any

from ..core.scalar import ScalarTensor
from ..core.dispatch import register_dispatch
from ..core.operator import Operator


class Add(Operator):
    ...

class Subtract(Operator):
    ...

class Multiply(Operator):
    ...

class Divide(Operator):
    ...

class Power(Operator):
    ...

class Exponential(Operator):
    ...

class Cosine(Operator):
    ...

class Sine(Operator):
    ...

@dataclass
class Constant(Operator):
    value: Any
    dtype: str

@register_dispatch()
def register_constant(op: Constant):
    c = ScalarTensor(op.value, op.dtype)
    return (c,)