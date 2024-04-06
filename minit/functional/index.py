import functools
from typing import Tuple
from ..core.tensor import Tensor
from ..core.dispatch import dispatch
from ..operator.index import Slice, SliceSet, Index, IndexSet, Split, Tie
from .utils import _convert_scalar


def slice(x: Tensor, start: Tensor, stop: Tensor, axis: int):
    start, stop = _convert_scalar(start, stop)
    (z,) = dispatch(Slice(axis=axis), x, start, stop)
    return z


def slice_set(x: Tensor, start: Tensor, stop: Tensor, axis: int, value: Tensor):
    start, stop = _convert_scalar(start, stop)
    (z,) = dispatch(SliceSet(axis=axis), x, start, stop, value)
    return z


def index(x: Tensor, index: Tensor, axis: int):
    (z,) = dispatch(Index(axis=axis), x, index)
    return z


def index_set(x: Tensor, index: Tensor, axis: int, value: Tensor):
    (z,) = dispatch(IndexSet(axis=axis), x, index, value)
    return z


def split(x: Tensor, axis: int, sizes: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
    sizes = _convert_scalar(*sizes)
    from .arith import constant
    zs = []
    offset = 0
    for size in sizes:
        offset += size.item()
    assert offset == x.shape[axis]
    offset = 0
    for size in sizes:
        start = offset
        stop = start + size.item()
        zs.append(slice(x, constant(start, dtype="int32"), constant(stop, dtype="int32"), axis))
        offset = stop
    return tuple(zs)


def tie(xs: Tuple[Tensor, ...], axis: int):
    (z,) = dispatch(Tie(axis=axis), *xs)
    return z
