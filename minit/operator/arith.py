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
class FloorDivide(Operator):
    ...

@dataclass
class Modulo(Operator):
    ...

@dataclass
class Power(Operator):
    ...

@dataclass
class Exponential(Operator):
    ...

@dataclass
class Logarithm(Operator):
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

@dataclass
class SelectMax(Operator):
    ...

@dataclass
class Select(Operator):
    ...

@register_dispatch()
def register_constant(op: Constant, *sizes: Tensor):
    assert not isinstance(op.value, Tensor), "cannot use tensor as constant"
    c = ScalarTensor(op.value, tuple(sizes), op.dtype)
    return (c,)

def register_elemwise_operator(op_type, op_py, dtype=None):
    @register_dispatch()
    def _register_elemwise_scalar(op: op_type, *args: ScalarTensor): # type: ignore
        items = [arg.item() for arg in args]
        c_item = op_py(*items)
        output_dtype = args[0].dtype if dtype is None else dtype(op, *(arg.dtype for arg in args))
        c = ScalarTensor(c_item, args[0].shape, output_dtype)
        return (c,)

    @register_dispatch(priority=-10, predicate=lambda *tys: any(ty is MetaTensor for ty in tys))
    def _register_elemwise_meta(op: op_type, *args: Tensor): # type: ignore
        output_dtype = args[0].dtype if dtype is None else dtype(op, *(arg.dtype for arg in args))
        c = MetaTensor(args[0].shape, output_dtype)
        return (c,)
    

def same_dtypes(op: Operator, *args: str) -> str:
    for arg in args[1:]:
        assert arg == args[0]
    return args[0]


def same_dtypes_return_bool(op: Operator, *args: str) -> str:
    for arg in args[1:]:
        assert arg == args[0]
    return "bool"


def same_dtypes_except_first(op: Operator, *args: str) -> str:
    for arg in args[2:]:
        assert arg == args[1]
    return args[1]


def dtype_from_operator(op: Operator, *args: Tensor) -> str:
    (arg,) = args
    return op.dtype


def register_elemwise_operators():
    for op_type, op_py, dtype in [
        (Add, lambda x, y: x + y, same_dtypes),
        (Subtract, lambda x, y: x - y, same_dtypes),
        (Multiply, lambda x, y: x * y, same_dtypes),
        (Divide,  lambda x, y: x / y, same_dtypes),
        (FloorDivide,  lambda x, y: x // y, same_dtypes),
        (Modulo, lambda x, y: x % y, same_dtypes),
        (Power, lambda x, y: pow(x, y), same_dtypes),
        (SelectMax, lambda x, y: max(x, y), same_dtypes),
        (Select, lambda condition, *args: args[condition], same_dtypes_except_first),
        (Sine, lambda x: math.sin(x), same_dtypes),
        (Cosine, lambda x: math.cos(x), same_dtypes),
        (Exponential, lambda x: math.exp(x), same_dtypes),
        (Logarithm, lambda x: math.log(x), same_dtypes),
        (GreaterThan, lambda x, y: x > y, same_dtypes_return_bool),
        (Equal, lambda x, y: x == y, same_dtypes_return_bool),
        (And, lambda x, y: bool(x) and bool(y), same_dtypes_return_bool),
        (Not, lambda x: not x, same_dtypes_return_bool),
        (Cast, lambda x: x, dtype_from_operator),
    ]:
        register_elemwise_operator(op_type, op_py, dtype)


register_elemwise_operators()
