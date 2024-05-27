from typing import Tuple

from ..operator.arith import Constant
from ..core.dispatch import dispatch
from ..core.tensor import Tensor


def _broadcast_constant(*args, dtype=None, shape=None) -> Tuple[Tensor, ...]:
    for arg in args:
        if isinstance(arg, Tensor):
            if shape is None:
                shape = arg.shape
            if dtype is None:
                dtype = arg.dtype
            # assert tensor.shape == arg.shape
            assert arg.dtype == dtype, f"{arg.dtype} vs {dtype}"
    def broadcast_scalar(scalar):
        assert dtype is not None
        (constant,) = dispatch(Constant(scalar, dtype), *shape)
        return constant
    return tuple([arg if isinstance(arg, Tensor) else broadcast_scalar(arg) for arg in args])


def _convert_constant(*args, dtype="int32") -> Tuple[Tensor, ...]:
    from .arith import constant
    for arg in args:
        if isinstance(arg, Tensor):
            assert arg.dtype == dtype
    return tuple([arg if isinstance(arg, Tensor) else constant(arg, dtype) for arg in args])
