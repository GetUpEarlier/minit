from typing import List, Optional, Tuple

from ..core.tensor import Tensor
from ..core.dispatch import dispatch
from ..operator.shape import AddAxis, Broadcast, Fold, Expand, Reinterpret, RemoveAxis, Transpose
from .utils import _convert_scalar


def fold(x: Tensor, start: int, stop: int) -> Tensor:
    (z,) = dispatch(Fold(start=start, stop=stop), x)
    return z


def expand(x: Tensor, axis: int, sizes: Tuple[Tensor, ...]) -> Tensor:
    sizes = _convert_scalar(*sizes)
    (z,) = dispatch(Expand(axis=axis), x, *sizes)
    return z


def add_axis(x: Tensor, axis: int, size: Optional[Tensor] = None) -> Tensor:
    if size is not None:
        (size,) = _convert_scalar(size)
    (z,) = dispatch(AddAxis(axis=axis), x)
    if size is not None:
        z = broadcast(z, axis, size)
    return z


def remove_axis(x: Tensor, axis: int) -> Tensor:
    (z,) = dispatch(RemoveAxis(axis=axis), x)
    return z


def broadcast(x: Tensor, axis: int, size: Tensor) -> Tensor:
    (size,) = _convert_scalar(size)
    (z,) = dispatch(Broadcast(axis=axis), x, size)
    return z


def transpose(x: Tensor, axis_a: int, axis_b: int) -> Tensor:
    if axis_a == axis_b:
        return x
    (z,) = dispatch(Transpose(axis_a, axis_b), x)
    return z


def repeat(x: Tensor, axis: int, size: Tensor) -> Tensor:
    (size,) = _convert_scalar(size)
    return fold(broadcast(add_axis(x, axis), axis, size), axis, axis+2)


def repeat_interleaved(x: Tensor, axis: int, size: Tensor) -> Tensor:
    (size,) = _convert_scalar(size)
    return fold(broadcast(add_axis(x, axis+1), axis+1, size), axis, axis+2)


def reinterpret(x: Tensor, target: str) -> Tensor:
    (z,) = dispatch(Reinterpret(target), x)
    return z
