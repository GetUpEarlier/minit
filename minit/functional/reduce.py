from ..core.dispatch import dispatch
from ..core.tensor import Tensor
from ..operator.reduce import Max, Sum


def sum(x: Tensor, axis: int) -> Tensor:
    (z,) = dispatch(Sum(axis=axis), x)
    return z


def mean(x: Tensor, axis: int) -> Tensor:
    from .arith import divide
    from .generate import fill
    size = x.shape[axis]
    return divide(sum(x, axis), fill(size, x.shape[:axis] + (1,) + x.shape[axis+1:], x.dtype))


def max(x: Tensor, axis: int) -> Tensor:
    (z,) = dispatch(Max(axis=axis), x)
    return z
