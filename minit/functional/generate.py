from numbers import Number
from typing import Tuple
from ..core.dispatch import dispatch
from ..operator.generate import Fill, GenerateInterval, GenerateSequence
from ..core.tensor import Tensor
from .utils import _convert_scalar


def generate_interval(start: Tensor, stop: Tensor, step: Tensor):
    start, stop, step = _convert_scalar(start, stop, step)
    (z,) = dispatch(GenerateInterval(), start, stop, step)
    return z


def generate_sequence(start: Tensor, size: Tensor, step: Tensor):
    start, size, step = _convert_scalar(start, size, step)
    (z,) = dispatch(GenerateSequence(), start, size, step)
    return z


def fill(value: Number, shape: Tuple[int, ...], dtype: str):
    from .arith import constant
    (z,) = dispatch(Fill(value=value), *map(constant, shape))
    return z
