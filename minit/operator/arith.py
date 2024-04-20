from dataclasses import dataclass
import math
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

def register_elemwise_operator(op_type, op_py):
    if op_py is not None:
        @register_dispatch()
        def _register_elemwise_scalar(op: op_type, *args: ScalarTensor): # type: ignore
            for arg in args:
                assert arg.dtype == args[0].dtype
            items = [arg.item() for arg in args]
            c_item = op_py(*items)
            c = ScalarTensor(c_item, args[0].dtype)
            return (c,)


def register_elemwise_operators():
    for op_type, op_py in [
        (Add, lambda x, y: x + y),
        (Subtract, lambda x, y: x - y),
        (Multiply, lambda x, y: x * y),
        (Divide,  lambda x, y: x / y),
        (Power, lambda x, y: pow(x, y)),
        (Sine, lambda x: math.sin(x)),
        (Cosine, lambda x: math.cos(x)),
        (Exponential, lambda x: math.exp(x)),
    ]:
        register_elemwise_operator(op_type, op_py)


register_elemwise_operators()
