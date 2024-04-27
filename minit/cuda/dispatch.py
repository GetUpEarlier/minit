import ctypes
import math
from typing import Union

from ..core.constant import ConstantTensor

from .kernel.cuda.cast import generate_cast_kernel

from ..core.shape import to_immediate_shape
from ..core.scalar import ScalarTensor
from ..core.tensor import Tensor
from .kernel.cuda.softmax import generate_softmax_kernel
from .kernel.cuda.rope import generate_rope_kernel
from .kernel.cuda.rms_norm import generate_rms_norm_kernel
from ..core.dtype import dtype_info
from .kernel.cuda.triangle import generate_triangle_kernel
from ..operator.special import RMSNorm, RoPE, Sigmoid, Softmax
from .kernel.cuda.transpose import generate_transpose_kernel
from .kernel.cuda.reduce import generate_reduce_kernel
from ..operator.reduce import Sum, Max
from .kernel.cuda.slice import generate_slice_kernel, generate_slice_set_kernel
from .kernel.cuda.index import generate_index_kernel
from ..operator.index import Index, Slice, Tie
from .kernel.cuda.broadcast import generate_broadcast_kernel
from ..operator.shape import AddAxis, Broadcast, Expand, Fold, Reinterpret, RemoveAxis, Transpose
from .kernel.cuda.fill import generate_fill_kernel
from .kernel.cuda.generate import generate_sequence_kernel
from ..operator.generate import Fill, GenerateSequence
from .kernel.cuda.elemwise import generate_elemwise_kernel
from ..operator.arith import Add, Constant, Cosine, Divide, Exponential, Power, Multiply, Sine, Subtract, Cast
from .tensor import CUDATensor
from ..operator.linalg import BatchMatrixMultiply, MatrixMultiply, TriangleLower, TriangleUpper
from ..core.dispatch import register_dispatch
from .kernel.cublas import generate_cublas_kernel
from .kernel.utils import get_cuda_dtype

import nvtx

def any_cuda_tensor(*args):
    for arg in args:
        if arg == CUDATensor:
            return True
    return False

def any_scalar_tensor(*args):
    for arg in args:
        if arg == ScalarTensor or arg == ConstantTensor:
            return True
    return False

def all_cuda_tensor(*args):
    for arg in args:
        if arg != CUDATensor:
            return False
    return True

@register_dispatch()
def dispatch_matmul(op: MatrixMultiply, a: CUDATensor, b: CUDATensor):
    m, k0 = a._shape
    n, k1 = b._shape
    assert k0 == k1
    assert a.dtype == b.dtype
    c = CUDATensor.allocate((m, n), a.dtype)
    kernel = generate_cublas_kernel("gemm", get_cuda_dtype(a.dtype))
    kernel(
        None,
        a.data_ptr,
        b.data_ptr,
        c.data_ptr,
        1,
        m,
        n,
        k0,
        None,
    )
    return (c,)


@register_dispatch()
def dispatch_batch_matmul(op: BatchMatrixMultiply, a: CUDATensor, b: CUDATensor):
    b0, m, k0 = a._shape
    b1, n, k1 = b._shape
    assert b0 == b1
    assert k0 == k1
    assert a.dtype == b.dtype
    c = CUDATensor.allocate((b0, m, n), a.dtype)
    kernel = generate_cublas_kernel("gemm", get_cuda_dtype(a.dtype))
    kernel(
        None,
        a.data_ptr,
        b.data_ptr,
        c.data_ptr,
        b0,
        m,
        n,
        k0,
        None,
    )
    return (c,)


def register_elemwise_operator(op_type, op_name, op_fan_in, op_expression, op_py):
    @register_dispatch()
    def _register_elemwise_cuda(op: op_type, *args: CUDATensor): # type: ignore
        for arg in args:
            assert arg._shape == () or arg._shape == args[0]._shape
            assert arg._dtype == args[0]._dtype
        c = CUDATensor.allocate(args[0]._shape, args[0]._dtype)
        kernel = generate_elemwise_kernel(op_name, op_fan_in, op_expression, get_cuda_dtype(args[0].dtype))
        size = 1
        for dim in args[0]._shape:
            size *= dim
        input_ptrs = (ctypes.c_void_p*op_fan_in)(*[arg.data_ptr for arg in args])
        strides = (ctypes.c_int*op_fan_in)(*[0 if len(arg._shape) == 0 else 1 for arg in args])
        kernel(None, input_ptrs, strides, c.data_ptr, size)
        return (c,)

    if op_py is not None:
        @register_dispatch(predicate=lambda *args: any_cuda_tensor(*args) and any_scalar_tensor(*args))
        def _register_elemwise_decay(op: op_type, *args: Union[CUDATensor, ScalarTensor, ConstantTensor]): # type: ignore
            from ..functional.generate import fill
            from ..core.dispatch import dispatch
            for arg in args:
                assert to_immediate_shape(arg.shape) == to_immediate_shape(args[0].shape)
                assert arg.dtype == args[0].dtype
            args = [arg if isinstance(arg, CUDATensor) else fill(arg.value(), arg.shape, arg.dtype) for arg in args]
            return dispatch(op, *args)


def register_elemwise_operators():

    for op_type, op_name, op_fan_in, op_expression, op_py in [
        (Add, "add", 2, "{0} + {1}", lambda x, y: x + y),
        (Subtract, "sub", 2, "{0} - {1}", lambda x, y: x - y),
        (Multiply, "mul", 2, "{0} * {1}", lambda x, y: x * y),
        (Divide, "div", 2, "{0} / {1}", lambda x, y: x / y),
        (Power, "exp", 2, "{1} == (T)2.0 ? {0} * {0} : {1} == (T)0.5 ? (T)sqrt((float){0}) : {1} == (T)-0.5 ? (T)rsqrt((float){0}) : (T)powf((float){0}, (float){1})", lambda x, y: pow(x, y)),
        (Sine, "sin", 1, "(T)sin((float){0})", lambda x: math.sin(x)),
        (Cosine, "cos", 1, "(T)cos((float){0})", lambda x: math.cos(x)),
        (Sigmoid, "sigmoid", 1, "{0} > (T)0 ? 1.0 / (1.0 + expf((float)-{0})) : (expf((float){0}) / (1.0 + expf((float){0})))", None),
        (Exponential, "exp", 1, "expf({0})", lambda x: math.exp(x)),
    ]:
        register_elemwise_operator(op_type, op_name, op_fan_in, op_expression, op_py)


register_elemwise_operators()


def register_reduce_operators():
    def register_reduce_operator(op_type, op_name, op_init, op_expr):
        @register_dispatch()
        def _register_reduce(op: op_type, x: CUDATensor): # type: ignore
            z = CUDATensor.allocate(x._shape[:op.axis] + (1,) + x._shape[op.axis+1:], x.dtype)
            kernel = generate_reduce_kernel(op_name, op_init, op_expr, get_cuda_dtype(x.dtype))
            a, b, c = 1, 1, 1
            for dim in x._shape[:op.axis]:
                a *= dim
            b = x._shape[op.axis]
            for dim in x._shape[op.axis+1:]:
                c *= dim
            kernel(None, x.data_ptr, z.data_ptr, a, b, c)
            return (z,)
    for op_type, op_name, op_init, op_expr in [
        (Sum, "sum", "0", "cub::Sum()"),
        (Max, "max", "-INFINITY", "cub::Max()"),
    ]:
        register_reduce_operator(op_type, op_name, op_init, op_expr)


register_reduce_operators()


@register_dispatch()
def register_fill(op: Fill, *sizes: Tensor):
    sizes = to_immediate_shape(sizes)
    size = 1
    for dim in sizes:
        size *= dim
    c = CUDATensor.allocate(tuple(sizes), op.dtype)
    kernel = generate_fill_kernel("fill", get_cuda_dtype(op.dtype))
    kernel(None, c.data_ptr, size, op.value)
    return (c,)


@register_dispatch()
def register_sequence(op: GenerateSequence, start: Tensor, size: Tensor, step: Tensor):
    size_value = size.item()
    assert start.dtype == step.dtype
    assert size.dtype == "int32"
    if not isinstance(start, CUDATensor):
        start = CUDATensor.from_item(start.item(), start.dtype)
    if not isinstance(step, CUDATensor):
        step = CUDATensor.from_item(step.item(), step.dtype)
    output = CUDATensor.allocate((size_value,), start.dtype)
    kernel = generate_sequence_kernel("sequence", get_cuda_dtype(start.dtype))
    kernel(
        None,
        output.data_ptr,
        start.data_ptr,
        step.data_ptr,
        size_value,
    )
    return (output,)


@register_dispatch()
def register_add_axis(op: AddAxis, x: CUDATensor):
    shape = x._shape[:op.axis] + (1,) + x._shape[op.axis:]
    z = CUDATensor.wrap(shape, x.dtype, x.memory)
    return (z,)


@register_dispatch()
def register_remove_axis(op: RemoveAxis, x: CUDATensor):
    assert x._shape[op.axis] == 1
    shape = x._shape[:op.axis] + x._shape[op.axis+1:]
    z = CUDATensor.wrap(shape, x.dtype, x.memory)
    return (z,)


@register_dispatch()
def register_fold(op: Fold, x: CUDATensor):
    size = 1
    for dim in x._shape[op.start:op.stop]:
        size *= dim
    shape = x._shape[:op.start] + (size,) + x._shape[op.stop:]
    z = CUDATensor.wrap(shape, x.dtype, x.memory)
    return (z,)


@register_dispatch()
def register_expand(op: Expand, x: CUDATensor, *sizes: Tensor):
    axis = op.axis
    shape = x._shape
    sizes = to_immediate_shape(sizes)
    size = 1
    for dim in sizes:
        size *= dim
    assert size == shape[axis]
    shape = shape[:axis] + sizes + shape[axis+1:]
    z = CUDATensor.wrap(shape, x._dtype, x._memory)
    return (z,)


@register_dispatch()
def register_broadcast(op: Broadcast, x: CUDATensor, size: Tensor):
    assert x._shape[op.axis] == 1
    a, c = 1, 1
    for dim in x._shape[:op.axis]:
        a *= dim
    b = size.item()
    for dim in x._shape[op.axis+1:]:
        c *= dim
    shape = x._shape[:op.axis] + (b,) + x._shape[op.axis+1:]
    z = CUDATensor.allocate(shape, x.dtype)
    kernel = generate_broadcast_kernel("broadcast", get_cuda_dtype(x.dtype))
    kernel(
        None,
        x.data_ptr,
        z.data_ptr,
        a,
        b,
        c,
    )
    return (z,)


@register_dispatch()
def register_index(op: Index, x: CUDATensor, index: CUDATensor):
    assert index.dtype == "int32"
    assert len(index._shape) == 1
    a, b, c, d = 1, 1, 1, 1
    for dim in x._shape[:op.axis]:
        a *= dim
    for dim in x._shape[op.axis+1:]:
        d *= dim
    b = x._shape[op.axis]
    (c,) = index._shape
    z = CUDATensor.allocate(x._shape[:op.axis] + (c,) + x._shape[op.axis+1:], x.dtype)
    kernel = generate_index_kernel("index", get_cuda_dtype(x.dtype))
    kernel(None,
           x.data_ptr,
           index.data_ptr,
           z.data_ptr,
           a,
           b,
           c,
           d,
    )
    return (z,)


@register_dispatch()
def register_slice(op: Slice, x: CUDATensor, start: Tensor, stop: Tensor):
    start = start.item()
    stop = stop.item()
    assert start >= 0
    assert start <= stop
    assert stop <= x._shape[op.axis]
    a, b, c = 1, 1, 1
    for dim in x._shape[:op.axis]:
        a *= dim
    for dim in x._shape[op.axis+1:]:
        c *= dim
    b = x._shape[op.axis]
    z = CUDATensor.allocate(x._shape[:op.axis] + ((stop - start),) + x._shape[op.axis+1:], x.dtype)
    kernel = generate_slice_kernel("slice", get_cuda_dtype(x.dtype))
    kernel(None,
           x.data_ptr,
           z.data_ptr,
           a,
           b,
           c,
           start,
           stop,
    )
    return (z,)


@register_dispatch()
def register_tie(op: Tie, *args: CUDATensor):
    for arg in args[1:]:
        assert arg.dtype == args[0].dtype
        assert arg._shape[:op.axis] == args[0]._shape[:op.axis]
        assert arg._shape[op.axis+1:] == args[0]._shape[op.axis+1:]
    a, b, c = 1, 0, 1
    for dim in args[0]._shape[:op.axis]:
        a *= dim
    for arg in args:
        b += arg._shape[op.axis]
    for dim in args[0]._shape[op.axis+1:]:
        c *= dim
    z = CUDATensor.allocate(args[0]._shape[:op.axis] + (b,) + args[0]._shape[op.axis+1:], args[0].dtype)
    kernel = generate_slice_set_kernel("slice_set", get_cuda_dtype(args[0].dtype))
    offset = 0
    for arg in args:
        start = offset
        stop = start + arg._shape[op.axis]
        kernel(None,
            arg.data_ptr,
            z.data_ptr,
            a,
            b,
            c,
            start,
            stop,
        )
        offset = stop
    return (z,)


@register_dispatch()
def register_transpose(op: Transpose, x: CUDATensor):
    assert op.axis_a < op.axis_b
    a, b, c, d, e = 1, 1, 1, 1, 1
    for dim in x._shape[:op.axis_a]:
        a *= dim
    b = x._shape[op.axis_a]
    for dim in x._shape[op.axis_a+1:op.axis_b]:
        c *= dim
    d = x._shape[op.axis_b]
    for dim in x._shape[op.axis_b+1:]:
        e *= dim
    shape = x._shape
    shape = shape[:op.axis_a] + (shape[op.axis_b],) + shape[op.axis_a+1:op.axis_b] + (shape[op.axis_a],) + shape[op.axis_b+1:]
    z = CUDATensor.allocate(shape, x.dtype)
    kernel = generate_transpose_kernel("transpose", get_cuda_dtype(x.dtype))
    kernel(
        None,
        x.data_ptr,
        z.data_ptr,
        a,
        b,
        c,
        d,
        e,
    )
    return (z,)


@register_dispatch()
def register_triangle_upper(op: TriangleUpper, x: CUDATensor, diagonal: Tensor):
    a, b, c, d = 1, 1, 1, 1
    for dim in x._shape[:-2]:
        a *= dim
    b = x._shape[-2]
    c = x._shape[-1]
    z = CUDATensor.allocate(x._shape, x.dtype)
    diagonal = diagonal.item()
    kernel = generate_triangle_kernel("triangle_upper", f"index[1] + diagonal <= index[2]", get_cuda_dtype(x.dtype))
    kernel(
        None,
        x.data_ptr,
        z.data_ptr,
        diagonal,
        a,
        b,
        c,
        d,
    )
    return (z,)


@register_dispatch()
def register_triangle_lower(op: TriangleLower, x: CUDATensor, diagonal: Tensor):
    a, b, c, d = 1, 1, 1, 1
    for dim in x._shape[:-2]:
        a *= dim
    b = x._shape[-2]
    c = x._shape[-1]
    z = CUDATensor.allocate(x._shape, x.dtype)
    diagonal = diagonal.item()
    kernel = generate_triangle_kernel("triangle_lower", f"index[1] > index[2] + diagonal", get_cuda_dtype(x.dtype))
    kernel(
        None,
        x.data_ptr,
        z.data_ptr,
        diagonal,
        a,
        b,
        c,
        d,
    )
    return (z,)


@register_dispatch()
def register_rms_norm_operator(op: RMSNorm, x: CUDATensor, weight: CUDATensor):
    z = CUDATensor.allocate(x._shape, x.dtype)
    kernel = generate_rms_norm_kernel("rms_norm", get_cuda_dtype(x.dtype))
    a, b, c = 1, 1, 1
    for dim in x._shape[:op.axis]:
        a *= dim
    b = x._shape[op.axis]
    for dim in x._shape[op.axis+1:]:
        c *= dim
    assert len(weight._shape) == 1
    assert weight._shape[0] == b
    kernel(None, x.data_ptr, weight.data_ptr, z.data_ptr, a, b, c, op.eps)
    return (z,)


@register_dispatch()
def register_rope_operator(op: RoPE, x: CUDATensor, freqs_cos: CUDATensor, freqs_sin: CUDATensor):
    bs, seqlen, nh, hs = x._shape
    z = CUDATensor.allocate(x._shape, x.dtype)
    kernel = generate_rope_kernel("rope", get_cuda_dtype(x.dtype))
    kernel(
        None,
        x.data_ptr,
        freqs_cos.data_ptr,
        freqs_sin.data_ptr,
        z.data_ptr,
        bs,
        seqlen,
        nh,
        hs,
    )
    return (z,)


@register_dispatch()
def register_softmax_operator(op: Softmax, x: CUDATensor):
    z = CUDATensor.allocate(x._shape, x.dtype)
    kernel = generate_softmax_kernel("softmax", get_cuda_dtype(x.dtype))
    a, b, c = 1, 1, 1
    for dim in x._shape[:op.axis]:
        a *= dim
    b = x._shape[op.axis]
    for dim in x._shape[op.axis+1:]:
        c *= dim
    kernel(None, x.data_ptr, z.data_ptr, a, b, c)
    return (z,)


@register_dispatch()
def register_reinterpret_operator(op: Reinterpret, x: CUDATensor):
    shape = x._shape
    source_size = dtype_info(x.dtype).size_in_bytes
    target_size = dtype_info(op.target).size_in_bytes
    last_dim = x.shape[-1].item()
    new_last_dim = last_dim * source_size // target_size
    assert new_last_dim * target_size // source_size == last_dim
    z = CUDATensor.wrap(shape[:-1] + (new_last_dim,), op.target, x._memory)
    return (z,)


@register_dispatch()
def register_cast_operator(op: Cast, x: CUDATensor): # type: ignore
    c = CUDATensor.allocate(x._shape, op.dtype)
    kernel = generate_cast_kernel(f"{x.dtype}_to_{c.dtype}", get_cuda_dtype(x.dtype), get_cuda_dtype(c.dtype))
    size = 1
    for dim in x._shape:
        size *= dim
    kernel(None, x.data_ptr, c.data_ptr, size)
    return (c,)
