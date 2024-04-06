import functools
from typing import Tuple
from ..core.tensor import Tensor
from ..core.dispatch import dispatch
from ..operator.index import Slice, SliceSet, Index, IndexSet, Split, Tie


def slice(x: Tensor, start: Tensor, stop: Tensor, axis: int):
    (z,) = dispatch(Slice(axis=axis), x, start, stop)
    return z


def slice_set(x: Tensor, start: Tensor, stop: Tensor, axis: int, value: Tensor):
    (z,) = dispatch(SliceSet(axis=axis), x, start, stop, value)
    return z


def index(x: Tensor, index: Tensor, axis: int):
    (z,) = dispatch(Index(axis=axis), x, index)
    return z


def index_set(x: Tensor, index: Tensor, axis: int, value: Tensor):
    (z,) = dispatch(IndexSet(axis=axis), x, index, value)
    return z


def split(x: Tensor, axis: int, sizes: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
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
        zs.append(slice(x, constant(start), constant(stop), axis))
        offset = stop
    return tuple(zs)


def tie(xs: Tuple[Tensor, ...], axis: int):
    (z,) = dispatch(Tie(axis=axis), *xs)
    return z
