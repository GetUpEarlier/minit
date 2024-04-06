from numbers import Number
from typing import Tuple
from ..core.dispatch import dispatch
from ..operator.generate import Fill, GenerateInterval, GenerateSequence
from ..core.tensor import Tensor


def generate_interval(start: Tensor, stop: Tensor, step: Tensor):
    (z,) = dispatch(GenerateInterval(), start, stop, step)
    return z


def generate_sequence(start: Tensor, size: Tensor, step: Tensor):
    (z,) = dispatch(GenerateSequence(), start, size, step)
    return z


def fill(value: Number, shape: Tuple[int, ...], dtype: str):
    from .arith import constant
    (z,) = dispatch(Fill(value=value), *map(constant, shape))
    return z
