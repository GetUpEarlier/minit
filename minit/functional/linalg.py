import functools

from ..core.tensor import Tensor
from ..core.dispatch import dispatch
from ..operator.linalg import MatrixMultiply, BatchMatrixMultiply, TriangleLower, TriangleUpper
from .utils import _convert_constant


def matrix_multiply(x: Tensor, y: Tensor):
    from .shape import fold, expand
    assert len(y.shape) == 2
    if len(x.shape) > 2:
        ms = x.shape[:len(x.shape)-1]
        x = fold(x, 0, len(ms))
    else:
        ms = None
    (z,) = dispatch(MatrixMultiply(), x, y)
    if ms is not None:
        z = expand(z, 0, _convert_constant(*ms))
    return z


def batch_matrix_multiply(x: Tensor, y: Tensor):
    from .shape import fold, expand
    assert len(x.shape) > 2
    assert len(y.shape) > 2
    bs = x.shape[:-2]
    # assert bs == y.shape[:-2]
    x = fold(x, 0, len(bs))
    y = fold(y, 0, len(bs))
    (z,) = dispatch(BatchMatrixMultiply(), x, y)
    z = expand(z, 0, _convert_constant(*bs))
    return z


def triangle_upper(x: Tensor, diagonal: Tensor):
    (diagonal,) = _convert_constant(diagonal)
    (z,) = dispatch(TriangleUpper(), x, diagonal)
    return z


def triangle_lower(x: Tensor, diagonal: Tensor):
    (diagonal,) = _convert_constant(diagonal)
    (z,) = dispatch(TriangleLower(), x, diagonal)
    return z
