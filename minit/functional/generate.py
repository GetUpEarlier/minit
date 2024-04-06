from numbers import Number
from typing import Tuple
from ..core.dispatch import dispatch
from ..operator.generate import Fill, GenerateInterval, GenerateSequence
from ..core.tensor import Tensor
from .utils import _convert_scalar


def generate_interval(start: Tensor, stop: Tensor, step: Tensor, *, dtype: str):
    start, stop, step = _convert_scalar(start, stop, step, dtype=dtype)
    (z,) = dispatch(GenerateInterval(), start, stop, step)
    return z


def generate_sequence(start: Tensor, size: Tensor, step: Tensor, *, dtype: str):
    start, step = _convert_scalar(start, step, dtype=dtype)
    (size,) = _convert_scalar(size)
    (z,) = dispatch(GenerateSequence(), start, size, step)
    return z


def fill(value: Number, shape: Tuple[Tensor, ...], dtype: str):
    shape = _convert_scalar(*shape)
    (z,) = dispatch(Fill(value=value, dtype=dtype), *shape)
    return z
