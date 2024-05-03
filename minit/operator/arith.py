from dataclasses import dataclass
import math
from typing import Any, Union

from ..core.tensor import Tensor
from ..core.scalar import ScalarTensor
from ..core.meta import MetaTensor
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

@dataclass
class GreaterThan(Operator):
    ...

@dataclass
class Equal(Operator):
    ...

@dataclass
class And(Operator):
    ...

@dataclass
class Not(Operator):
    ...

@register_dispatch()
def register_constant(op: Constant, *sizes: Tensor):
    assert not isinstance(op.value, Tensor), "cannot use tensor as constant"
    c = ScalarTensor(op.value, tuple(sizes), op.dtype)
    return (c,)

def register_elemwise_operator(op_type, op_py, dtype=None):
    @register_dispatch()
    def _register_elemwise_scalar(op: op_type, *args: ScalarTensor): # type: ignore
        for arg in args:
            assert arg.dtype == args[0].dtype
        items = [arg.item() for arg in args]
        c_item = op_py(*items)
        output_dtype = args[0].dtype if dtype is None else dtype(op, *(arg.dtype for arg in args))
        c = ScalarTensor(c_item, args[0].shape, output_dtype)
        return (c,)

    @register_dispatch(priority=-1)
    def _register_elemwise_meta(op: op_type, *args: Tensor): # type: ignore
        for arg in args:
            assert arg.dtype == args[0].dtype
        output_dtype = args[0].dtype if dtype is None else dtype(op, *(arg.dtype for arg in args))
        c = MetaTensor(args[0].shape, output_dtype)
        return (c,)


def register_elemwise_operators():
    for op_type, op_py, dtype in [
        (Add, lambda x, y: x + y, None),
        (Subtract, lambda x, y: x - y, None),
        (Multiply, lambda x, y: x * y, None),
        (Divide,  lambda x, y: x / y, None),
        (Power, lambda x, y: pow(x, y), None),
        (Sine, lambda x: math.sin(x), None),
        (Cosine, lambda x: math.cos(x), None),
        (Exponential, lambda x: math.exp(x), None),
        (GreaterThan, lambda x, y: x > y, lambda op, *args: "bool"),
        (Equal, lambda x, y: x == y, lambda op, *args: "bool"),
        (And, lambda x, y: bool(x) and bool(y), lambda op, *args: "bool"),
        (Not, lambda x: not x, lambda op, *args: "bool"),
        (Cast, lambda x: x, lambda op, *args: op.dtype),
    ]:
        register_elemwise_operator(op_type, op_py, dtype)


register_elemwise_operators()
