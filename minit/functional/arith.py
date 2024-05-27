from numbers import Number
from typing import Optional, Union
from ..core.tensor import Tensor
from ..core.dispatch import dispatch
from ..operator.arith import Add, And, Cast, Constant, Cosine, Equal, Exponential, FloorDivide, GreaterThan, Logarithm, Modulo, Not, Power, Select, SelectMax, Sine, Subtract, Multiply, Divide
from .utils import _broadcast_constant


def add(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(Add(), x, y)
    return z


def subtract(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(Subtract(), x, y)
    return z


def multiply(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(Multiply(), x, y)
    return z


def divide(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(Divide(), x, y)
    return z


def floor_divide(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(FloorDivide(), x, y)
    return z


def modulo(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(Modulo(), x, y)
    return z


def power(base: Tensor, exponent: Tensor):
    base, exponent = _broadcast_constant(base, exponent)
    (z,) = dispatch(Power(), base, exponent)
    return z


def exponential(x: Tensor):
    (z,) = dispatch(Exponential(), x)
    return z


def logarithm(x: Tensor):
    (z,) = dispatch(Logarithm(), x)
    return z


def square(x: Tensor):
    from .generate import fill
    return power(x, fill(2, x.shape, x.dtype))


def square_root(x: Tensor):
    from .generate import fill
    return power(x, fill(1/2, x.shape, x.dtype))


def sine(x: Tensor):
    (z,) = dispatch(Sine(), x)
    return z


def cosine(x: Tensor):
    (z,) = dispatch(Cosine(), x)
    return z


def constant(x: Number, dtype: str):
    opr = Constant(value=x, dtype=dtype)
    (z,) = dispatch(opr)
    assert isinstance(z, Tensor)
    return z


def cast(x: Tensor, dtype: str):
    if x.dtype == dtype:
        return x
    (z,) = dispatch(Cast(dtype), x)
    return z


def greater_than(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(GreaterThan(), x, y)
    return z


def equal(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(Equal(), x, y)
    return z


def not_equal(x: Tensor, y: Tensor):
    return logical_not(equal(x, y))


def less_than(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    return logical_and(logical_not(greater_than(x, y)), logical_not(equal(x, y)))


def logical_and(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(And(), x, y)
    return z


def logical_not(x: Tensor):
    (x,) = _broadcast_constant(x)
    (z,) = dispatch(Not(), x)
    return z


def logical_or(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    return logical_and(logical_not(x), logical_not(y))


def greater_equal(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    return logical_or(greater_than(x, y), equal(x, y))


def less_equal(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    return logical_not(greater_than(x, y))


def select_max(x: Tensor, y: Tensor):
    x, y = _broadcast_constant(x, y)
    (z,) = dispatch(SelectMax(), x, y)
    return z


def select(index: Tensor, *args: Tensor, dtype: Optional[str]=None):
    args = _broadcast_constant(*args, shape=index.shape, dtype=dtype)
    (z,) = dispatch(Select(), index, *args)
    return z


def where(condition: Tensor, true_value: Tensor, false_value: Tensor, *, dtype: Optional[str]=None):
    assert condition.dtype == "bool"
    return select(condition, false_value, true_value, dtype=dtype)
