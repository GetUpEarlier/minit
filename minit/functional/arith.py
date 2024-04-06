from numbers import Number
from ..core.tensor import Tensor
from ..core.dispatch import dispatch
from ..operator.arith import Add, Constant, Cosine, Exponential, Power, Sine, Subtract, Multiply, Divide


def add(x: Tensor, y: Tensor):
    (z,) = dispatch(Add(), x, y)
    return z


def subtract(x: Tensor, y: Tensor):
    (z,) = dispatch(Subtract(), x, y)
    return z


def multiply(x: Tensor, y: Tensor):
    (z,) = dispatch(Multiply(), x, y)
    return z


def divide(x: Tensor, y: Tensor):
    (z,) = dispatch(Divide(), x, y)
    return z


def power(base: Tensor, exponent: Tensor):
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


def constant(x: Number):
    opr = Constant()
    opr.value = x
    (z,) = dispatch(opr)
    return z
