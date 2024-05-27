from numbers import Number
from typing import Optional, Sequence, Tuple, Union
from ..core.dispatch import dispatch
from ..operator.generate import Fill, GenerateInterval, GenerateSequence
from ..core.tensor import Tensor
from .utils import _broadcast_constant, _convert_constant


def generate_interval(start: Union[Tensor, Number], stop: Union[Tensor, Number], step: Union[Tensor, Number]=1, *, dtype: str):
    start, stop, step = _convert_constant(start, stop, step, dtype=dtype)
    (z,) = dispatch(GenerateInterval(), start, stop, step)
    return z


def generate_sequence(start: Union[Tensor, Number], size: Union[Tensor, Number], step: Union[Tensor, Number]=1, *, dtype: str):
    start, step = _convert_constant(start, step, dtype=dtype)
    (size,) = _convert_constant(size)
    (z,) = dispatch(GenerateSequence(), start, size, step)
    return z


def fill(value: Union[Number, Tensor], shape: Sequence[Tensor], dtype: Optional[str]=None):
    shape = _convert_constant(*shape)
    (value,) = _broadcast_constant(value, dtype=dtype, shape=())
    if dtype is not None:
        assert dtype == value.dtype
    assert len(value.shape) == 0
    (z,) = dispatch(Fill(), value, *shape)
    return z
