from typing import Tuple
from ..core.tensor import Tensor


def _broadcast_scalar(*args) -> Tuple[Tensor, ...]:
    from .generate import fill
    results = []
    tensor = None
    for arg in args:
        if isinstance(arg, Tensor):
            if tensor is None:
                tensor = arg
            else:
                # assert tensor.shape == arg.shape
                assert tensor.dtype == arg.dtype
    for arg in args:
        if not isinstance(arg, Tensor):
            arg = fill(arg, tensor.shape, tensor.dtype)
        results.append(arg)
    return tuple(results)


def _convert_scalar(*args, dtype="int32") -> Tuple[Tensor, ...]:
    from .arith import constant
    results = []
    for arg in args:
        if not isinstance(arg, Tensor):
            arg = constant(arg, dtype)
        results.append(arg)
    return tuple(results)
