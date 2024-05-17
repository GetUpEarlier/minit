from typing import List, Optional, Tuple

from ..core.scalar import ScalarTensor

from ..core.tensor import Tensor
from ..core.dispatch import dispatch, register_dispatch
from ..operator.shape import AddAxis, Broadcast, Fold, Expand, Reinterpret, RemoveAxis, Transpose
from .utils import _convert_constant


def fold(x: Tensor, start: int, stop: int) -> Tensor:
    if stop - start == 1:
        return x
    (z,) = dispatch(Fold(start=start, stop=stop), x)
    return z


def expand(x: Tensor, axis: int, sizes: Tuple[Tensor, ...]) -> Tensor:
    if len(sizes) == 1:
        # TODO: assertion
        return x
    sizes = _convert_constant(*sizes)
    (z,) = dispatch(Expand(axis=axis), x, *sizes)
    return z


def add_axis(x: Tensor, axis: int, size: Optional[Tensor] = None) -> Tensor:
    if size is not None:
        (size,) = _convert_constant(size)
    (z,) = dispatch(AddAxis(axis=axis), x)
    if size is not None:
        z = broadcast(z, axis, size)
    return z


def remove_axis(x: Tensor, axis: int) -> Tensor:
    (z,) = dispatch(RemoveAxis(axis=axis), x)
    return z


def broadcast(x: Tensor, axis: int, size: Tensor) -> Tensor:
    (size,) = _convert_constant(size)
    assert axis < len(x.shape)
    (z,) = dispatch(Broadcast(axis=axis), x, size)
    return z


def transpose(x: Tensor, axis_a: int, axis_b: int) -> Tensor:
    if axis_a == axis_b:
        return x
    if axis_a > axis_b:
        axis_a, axis_b = axis_b, axis_a
    (z,) = dispatch(Transpose(axis_a, axis_b), x)
    return z


def repeat(x: Tensor, axis: int, size: Tensor) -> Tensor:
    (size,) = _convert_constant(size)
    return fold(broadcast(add_axis(x, axis), axis, size), axis, axis+2)


def repeat_interleaved(x: Tensor, axis: int, size: Tensor) -> Tensor:
    (size,) = _convert_constant(size)
    return fold(broadcast(add_axis(x, axis+1), axis+1, size), axis, axis+2)


def reinterpret(x: Tensor, target: str) -> Tensor:
    (z,) = dispatch(Reinterpret(target), x)
    return z


@register_dispatch()
def dispatch_add_axis(op: AddAxis, x: ScalarTensor):
    shape = x._shape[:op.axis] + (ScalarTensor(1, (), "int32"),) + x._shape[op.axis:]
    z = ScalarTensor(x._value, shape, x._dtype)
    return (z,)
