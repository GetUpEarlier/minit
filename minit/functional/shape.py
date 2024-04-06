from typing import List, Optional
from ..core.tensor import Tensor
from ..core.dispatch import dispatch
from ..operator.shape import AddAxis, Broadcast, Fold, Expand, RemoveAxis, Transpose


def fold(x: Tensor, start: int, stop: int) -> Tensor:
    (z,) = dispatch(Fold(start=start, stop=stop), x)
    return z


def expand(x: Tensor, axis: int, sizes: List[Tensor]) -> Tensor:
    (z,) = dispatch(Expand(axis=axis), x, *sizes)
    return z


def add_axis(x: Tensor, axis: int, size: Optional[Tensor] = None) -> Tensor:
    (z,) = dispatch(AddAxis(axis=axis), x)
    if size is not None:
        z = broadcast(z, axis, size)
    return z


def remove_axis(x: Tensor, axis: int) -> Tensor:
    (z,) = dispatch(RemoveAxis(axis=axis), x)
    return z


def broadcast(x: Tensor, axis: int, size: Tensor) -> Tensor:
    (z,) = dispatch(Broadcast(axis=axis), x, size)
    return z


def transpose(x: Tensor, axis_a: int, axis_b: int) -> Tensor:
    (z,) = dispatch(Transpose(axis_a, axis_b), x)
    return z


def repeat(x: Tensor, axis: int, size: Tensor) -> Tensor:
    return fold(broadcast(add_axis(x, axis), axis, size), axis, axis+2)


def repeat_interleaved(x: Tensor, axis: int, size: Tensor) -> Tensor:
    return fold(broadcast(add_axis(x, axis+1), axis+1, size), axis, axis+2)
