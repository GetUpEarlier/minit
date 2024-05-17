from typing import Union
from ..core.scalar import ScalarTensor
from ..core.meta import MetaTensor
from ..core.dispatch import dispatch, register_dispatch
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


@register_dispatch()
def dispatch_max(op: Union[Max, Sum], x: MetaTensor):
    shape = x.shape[:op.axis] + (ScalarTensor(1, (), "int32"),) + x.shape[op.axis+1:]
    z = MetaTensor(shape, x.dtype)
    return (z,)
