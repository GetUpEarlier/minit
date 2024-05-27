from dataclasses import dataclass

from ..core.tensor import Tensor
from ..core.meta import MetaTensor
from ..core.dispatch import register_dispatch
from ..core.operator import Operator


@dataclass
class MatrixMultiply(Operator):
    ...


@dataclass
class BatchMatrixMultiply(Operator):
    ...


@dataclass
class TriangleUpper(Operator):
    pass


@dataclass
class TriangleLower(Operator):
    diagonal: int


@register_dispatch()
def dispatch_matrix_multiply(op: MatrixMultiply, x: MetaTensor, y: MetaTensor):
    assert x.dtype == y.dtype
    m, k0 = x.shape
    n, k1 = y.shape
    z = MetaTensor((m, n), x.dtype)
    return (z,)


@register_dispatch()
def dispatch_triu(op: TriangleUpper, x: MetaTensor, y: Tensor):
    return (x,)
