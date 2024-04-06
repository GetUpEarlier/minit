from numbers import Number
from typing import Union
from ..core.tensor import Tensor
from ..core.dispatch import dispatch
from ..operator.arith import Add, Constant, Cosine, Exponential, Power, Sine, Subtract, Multiply, Divide
from .utils import _broadcast_scalar


def add(x: Tensor, y: Tensor):
    x, y = _broadcast_scalar(x, y)
    (z,) = dispatch(Add(), x, y)
    return z


def subtract(x: Tensor, y: Tensor):
    x, y = _broadcast_scalar(x, y)
    (z,) = dispatch(Subtract(), x, y)
    return z


def multiply(x: Tensor, y: Tensor):
    x, y = _broadcast_scalar(x, y)
    (z,) = dispatch(Multiply(), x, y)
    return z


def divide(x: Tensor, y: Tensor):
    x, y = _broadcast_scalar(x, y)
    (z,) = dispatch(Divide(), x, y)
    return z


def power(base: Tensor, exponent: Tensor):
    base, exponent = _broadcast_scalar(base, exponent)
    (z,) = dispatch(Power(), base, exponent)
    return z


def exponential(x: Tensor):
    (z,) = dispatch(Exponential(), x)
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
    return z
