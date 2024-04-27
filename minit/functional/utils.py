from typing import Tuple

from ..operator.arith import Constant
from ..core.dispatch import dispatch
from ..core.tensor import Tensor


def _broadcast_constant(*args) -> Tuple[Tensor, ...]:
    tensor = None
    for arg in args:
        if isinstance(arg, Tensor):
            if tensor is None:
                tensor = arg
            else:
                # assert tensor.shape == arg.shape
                assert tensor.dtype == arg.dtype
    shape = None
    def broadcast_scalar(scalar):
        nonlocal shape
        (constant,) = dispatch(Constant(scalar, tensor.dtype), *tensor.shape)
        return constant
    return tuple([arg if isinstance(arg, Tensor) else broadcast_scalar(arg) for arg in args])


def _convert_constant(*args, dtype="int32") -> Tuple[Tensor, ...]:
    from .arith import constant
    return tuple([arg if isinstance(arg, Tensor) else constant(arg, dtype) for arg in args])
