from typing import Tuple

from ..core.scalar import ScalarTensor
from ..core.tensor import Tensor


def _scalar_to_tensor(*args) -> Tuple[Tensor, ...]:
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
    return tuple([arg if isinstance(arg, Tensor) else fill(arg, (), tensor.dtype) for arg in args])


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
    shape = None
    def broadcast_scalar(scalar):
        nonlocal shape
        if shape is None:
            shape = tensor.shape
        if len(shape) == 0:
            return ScalarTensor(scalar, tensor.dtype)
        else:
            return fill(scalar, tensor.shape, tensor.dtype)
    return tuple([arg if isinstance(arg, Tensor) else broadcast_scalar(arg) for arg in args])


def _convert_scalar(*args, dtype="int32") -> Tuple[Tensor, ...]:
    from .arith import constant
    return tuple([arg if isinstance(arg, Tensor) else constant(arg, dtype) for arg in args])
