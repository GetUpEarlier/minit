from .operator import Dequantize
from ..core.tensor import Tensor
from .tensor import QuantizedTensor
from ..operator.linalg import MatrixMultiply
from ..core.dispatch import dispatch, register_dispatch


@register_dispatch()
def dispatch_quantized_matrix_multiply(op: MatrixMultiply, a: Tensor, b: QuantizedTensor):
    b = b.dequantize()
    return dispatch(op, a, b)


@register_dispatch()
def dispatch_dequantize(op: Dequantize, data: Tensor, group: Tensor, zero: Tensor, scale: Tensor):
    scale = scale.index(group, axis=op.axis)
    zero = zero.index(group, axis=op.axis)
    dtype = scale.dtype
    output = (data.cast(dtype) - (zero.cast(dtype)+1)) * scale
    return (output,)
