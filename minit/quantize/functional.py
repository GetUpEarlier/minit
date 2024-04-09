from ..core.dispatch import dispatch
from .operator import Dequantize
from ..core.tensor import Tensor


def dequantize(x: Tensor, group: Tensor, zero: Tensor, scale: Tensor, axis: int) -> Tensor:
    (z,) = dispatch(Dequantize(axis), x, group, zero, scale)
    return z
