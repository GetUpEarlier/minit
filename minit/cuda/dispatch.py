import ctypes

from .kernel.cuda.triangle import generate_triangle_kernel

from ..operator.special import Sigmoid

from .kernel.cuda.transpose import generate_transpose_kernel

from .kernel.cuda.reduce import generate_reduce_kernel
from ..operator.reduce import Sum, Max

from .kernel.cuda.slice import generate_slice_kernel, generate_slice_set_kernel

from .kernel.cuda.index import generate_index_kernel

from ..operator.index import Index, Slice, Tie

from .kernel.cuda.broadcast import generate_broadcast_kernel
from ..operator.shape import AddAxis, Broadcast, Expand, Fold, RemoveAxis, Transpose

from .kernel.cuda.fill import generate_fill_kernel
from .kernel.cuda.generate import generate_sequence_kernel
from ..operator.generate import Fill, GenerateSequence
from .kernel.cuda.elemwise import generate_elemwise_kernel
from ..operator.arith import Add, Constant, Cosine, Divide, Exponential, Power, Multiply, Sine, Subtract
from .tensor import CUDATensor
from ..operator.linalg import BatchMatrixMultiply, MatrixMultiply, TriangleLower, TriangleUpper
from ..core.dispatch import register
from .kernel.cublas import generate_cublas_kernel

@register
def dispatch_matmul(op: MatrixMultiply, a: CUDATensor, b: CUDATensor):
    m, k0 = a.shape
    n, k1 = b.shape
    assert k0 == k1
    c = CUDATensor.allocate((m, n), a.dtype)
    kernel = generate_cublas_kernel("gemm")
    kernel(
        None,
        ctypes.c_void_p(a.data_ptr),
        ctypes.c_void_p(b.data_ptr),
        ctypes.c_void_p(c.data_ptr),
        ctypes.c_size_t(1),
        ctypes.c_size_t(m),
        ctypes.c_size_t(n),
        ctypes.c_size_t(k0),
        None,
    )
    return (c,)


@register
def dispatch_batch_matmul(op: BatchMatrixMultiply, a: CUDATensor, b: CUDATensor):
    b0, m, k0 = a.shape
    b1, n, k1 = b.shape
    assert b0 == b1
    assert k0 == k1
    c = CUDATensor.allocate((b0, m, n), a.dtype)
    kernel = generate_cublas_kernel("gemm")
    kernel(
        None,
        ctypes.c_void_p(a.data_ptr),
        ctypes.c_void_p(b.data_ptr),
        ctypes.c_void_p(c.data_ptr),
        ctypes.c_size_t(b0),
        ctypes.c_size_t(m),
        ctypes.c_size_t(n),
        ctypes.c_size_t(k0),
        None,
    )
    return (c,)


def register_elemwise_operators():
    def register_elemwise_operator(op_type, op_name, op_fan_in, op_expression):
        @register
        def _register_elemwise(op: op_type, *args: CUDATensor): # type: ignore
            for arg in args:
                assert arg.shape == args[0].shape
            c = CUDATensor.allocate(args[0].shape, args[0].dtype)
            kernel = generate_elemwise_kernel(op_name, op_fan_in, op_expression)
            size = 1
            for dim in args[0].shape:
                size *= dim
            input_ptrs = (ctypes.c_void_p*op_fan_in)(*[arg.data_ptr for arg in args])
            kernel(None, input_ptrs, ctypes.c_void_p(c.data_ptr), ctypes.c_size_t(size))
            return (c,)
    for op_type, op_name, op_fan_in, op_expression in [
        (Add, "add", 2, "values[0] + values[1]"),
        (Subtract, "sub", 2, "values[0] - values[1]"),
        (Multiply, "mul", 2, "values[0] * values[1]"),
        (Divide, "div", 2, "values[0] / values[1]"),
        (Power, "exp", 2, "values[1] == 2.0 ? values[0] * values[0] : values[1] == 0.5 ? sqrt(values[0]) : values[1] == -0.5 ? rsqrt(values[0]) : powf(values[0], values[1])"),
        (Sine, "sin", 1, "sin(values[0])"),
        (Cosine, "cos", 1, "cos(values[0])"),
        (Sigmoid, "sigmoid", 1, "values[0] > 0 ? 1.0 / (1.0 + expf(-values[0])) : (expf(values[0]) / (1.0 + expf(values[0])))"),
        (Exponential, "exp", 1, "expf(values[0])"),
    ]:
        register_elemwise_operator(op_type, op_name, op_fan_in, op_expression)


register_elemwise_operators()


def register_reduce_operators():
    def register_reduce_operator(op_type, op_name, op_init, op_expr):
        @register
        def _register_reduce(op: op_type, x: CUDATensor): # type: ignore
            z = CUDATensor.allocate(x.shape[:op.axis] + (1,) + x.shape[op.axis+1:], x.dtype)
            kernel = generate_reduce_kernel(op_name, op_init, op_expr)
            a, b, c = 1, 1, 1
            for dim in x.shape[:op.axis]:
                a *= dim
            b = x.shape[op.axis]
            for dim in x.shape[op.axis+1:]:
                c *= dim
            kernel(None, ctypes.c_void_p(x.data_ptr), ctypes.c_void_p(z.data_ptr), ctypes.c_size_t(a), ctypes.c_size_t(b), ctypes.c_size_t(c))
            return (z,)
    for op_type, op_name, op_init, op_expr in [
        (Sum, "sum", "0", "x + y"),
        (Max, "max", "-INFINITY", "max(x, y)"),
    ]:
        register_reduce_operator(op_type, op_name, op_init, op_expr)


register_reduce_operators()


@register
def register_constant(op: Constant):
    c = CUDATensor.from_item(op.value)
    return (c,)


@register
def register_fill(op: Fill, *sizes: CUDATensor):
    sizes = list(map(lambda x: x.item(), sizes))
    size = 1
    for dim in sizes:
        size *= dim
    c = CUDATensor.allocate(tuple(sizes), "float32")
    kernel = generate_fill_kernel("fill")
    kernel(None, ctypes.c_void_p(c.data_ptr), ctypes.c_size_t(size), ctypes.c_float(op.value))
    return (c,)


@register
def register_sequence(op: GenerateSequence, start: CUDATensor, size: CUDATensor, step: CUDATensor):
    size_value = size.item()
    output = CUDATensor.allocate((size_value,), "float32")
    kernel = generate_sequence_kernel("sequence")
    kernel(
        None,
        ctypes.c_void_p(output.data_ptr),
        ctypes.c_void_p(start.data_ptr),
        ctypes.c_void_p(step.data_ptr),
        ctypes.c_size_t(size_value),
    )
    return (output,)


@register
def register_add_axis(op: AddAxis, x: CUDATensor):
    shape = x.shape[:op.axis] + (1,) + x.shape[op.axis:]
    z = CUDATensor.wrap(shape, x.dtype, x.memory)
    return (z,)


@register
def register_remove_axis(op: RemoveAxis, x: CUDATensor):
    assert x.shape[op.axis] == 1
    shape = x.shape[:op.axis] + x.shape[op.axis+1:]
    z = CUDATensor.wrap(shape, x.dtype, x.memory)
    return (z,)


@register
def register_fold(op: Fold, x: CUDATensor):
    size = 1
    for dim in x.shape[op.start:op.stop]:
        size *= dim
    shape = x.shape[:op.start] + (size,) + x.shape[op.stop:]
    z = CUDATensor.wrap(shape, x.dtype, x.memory)
    return (z,)


@register
def register_expand(op: Expand, x: CUDATensor, *sizes: CUDATensor):
    def get_int(size: CUDATensor):
        return size.item()
    sizes = tuple(map(get_int, sizes))
    size = 1
    for dim in sizes:
        size *= dim
    assert size == x.shape[op.axis]
    shape = x.shape[:op.axis] + sizes + x.shape[op.axis+1:]
    z = CUDATensor.wrap(shape, x.dtype, x.memory)
    return (z,)


@register
def register_broadcast(op: Broadcast, x: CUDATensor, size: CUDATensor):
    assert x.shape[op.axis] == 1
    a, c = 1, 1
    for dim in x.shape[:op.axis]:
        a *= dim
    b = size.item()
    for dim in x.shape[op.axis+1:]:
        c *= dim
    shape = x.shape[:op.axis] + (b,) + x.shape[op.axis+1:]
    z = CUDATensor.allocate(shape, x.dtype)
    kernel = generate_broadcast_kernel("broadcast")
    kernel(
        None,
        ctypes.c_void_p(x.data_ptr),
        ctypes.c_void_p(z.data_ptr),
        ctypes.c_size_t(a),
        ctypes.c_size_t(b),
        ctypes.c_size_t(c),
    )
    return (z,)


@register
def register_index(op: Index, x: CUDATensor, index: CUDATensor):
    # assert len(index.shape) == len(weight.shape)
    assert len(index.shape) == 1
    a, b, c, d = 1, 1, 1, 1
    for dim in x.shape[:op.axis]:
        a *= dim
    for dim in x.shape[op.axis+1:]:
        d *= dim
    b = x.shape[op.axis]
    (c,) = index.shape
    z = CUDATensor.allocate(x.shape[:op.axis] + (c,) + x.shape[op.axis+1:], x.dtype)
    kernel = generate_index_kernel("index")
    kernel(None,
           ctypes.c_void_p(x.data_ptr),
           ctypes.c_void_p(index.data_ptr),
           ctypes.c_void_p(z.data_ptr),
           ctypes.c_size_t(a),
           ctypes.c_size_t(b),
           ctypes.c_size_t(c),
           ctypes.c_size_t(d)
    )
    return (z,)


@register
def register_slice(op: Slice, x: CUDATensor, start: CUDATensor, stop: CUDATensor):
    start = start.item()
    stop = stop.item()
    assert start >= 0
    assert start <= stop
    assert stop <= x.shape[op.axis]
    a, b, c = 1, 1, 1
    for dim in x.shape[:op.axis]:
        a *= dim
    for dim in x.shape[op.axis+1:]:
        c *= dim
    b = x.shape[op.axis]
    z = CUDATensor.allocate(x.shape[:op.axis] + ((stop - start),) + x.shape[op.axis+1:], x.dtype)
    kernel = generate_slice_kernel("slice")
    kernel(None,
           ctypes.c_void_p(x.data_ptr),
           ctypes.c_void_p(z.data_ptr),
           ctypes.c_size_t(a),
           ctypes.c_size_t(b),
           ctypes.c_size_t(c),
           ctypes.c_size_t(start),
           ctypes.c_size_t(stop),
    )
    return (z,)


@register
def register_tie(op: Tie, *args: CUDATensor):
    a, b, c = 1, 0, 1
    for dim in args[0].shape[:op.axis]:
        a *= dim
    for arg in args:
        b += arg.shape[op.axis]
    for dim in args[0].shape[op.axis+1:]:
        c *= dim
    z = CUDATensor.allocate(args[0].shape[:op.axis] + (b,) + args[0].shape[op.axis+1:], args[0].dtype)
    kernel = generate_slice_set_kernel("slice_set")
    offset = 0
    for arg in args:
        start = offset
        stop = start + arg.shape[op.axis]
        kernel(None,
            ctypes.c_void_p(arg.data_ptr),
            ctypes.c_void_p(z.data_ptr),
            ctypes.c_size_t(a),
            ctypes.c_size_t(b),
            ctypes.c_size_t(c),
            ctypes.c_size_t(start),
            ctypes.c_size_t(stop),
        )
        offset = stop
    return (z,)


@register
def register_transpose(op: Transpose, x: CUDATensor):
    assert op.axis_a < op.axis_b
    a, b, c, d, e = 1, 1, 1, 1, 1
    for dim in x.shape[:op.axis_a]:
        a *= dim
    b = x.shape[op.axis_a]
    for dim in x.shape[op.axis_a+1:op.axis_b]:
        c *= dim
    d = x.shape[op.axis_b]
    for dim in x.shape[op.axis_b+1:]:
        e *= dim
    shape = x.shape
    shape = shape[:op.axis_a] + (shape[op.axis_b],) + shape[op.axis_a+1:op.axis_b] + (shape[op.axis_a],) + shape[op.axis_b+1:]
    z = CUDATensor.allocate(shape, x.dtype)
    kernel = generate_transpose_kernel("transpose")
    kernel(
        None,
        ctypes.c_void_p(x.data_ptr),
        ctypes.c_void_p(z.data_ptr),
        ctypes.c_size_t(a),
        ctypes.c_size_t(b),
        ctypes.c_size_t(c),
        ctypes.c_size_t(d),
        ctypes.c_size_t(e),
    )
    return (z,)


@register
def register_triangle_upper(op: TriangleUpper, x: CUDATensor):
    a, b, c, d = 1, 1, 1, 1
    for dim in x.shape[:-2]:
        a *= dim
    b = x.shape[-2]
    c = x.shape[-1]
    assert b == c
    z = CUDATensor.allocate(x.shape, x.dtype)
    diagonal = op.diagonal
    kernel = generate_triangle_kernel("triangle_upper", f"index[1] + {diagonal} <= index[2]")
    kernel(
        None,
        ctypes.c_void_p(x.data_ptr),
        ctypes.c_void_p(z.data_ptr),
        ctypes.c_size_t(a),
        ctypes.c_size_t(b),
        ctypes.c_size_t(c),
        ctypes.c_size_t(d),
    )
    return (z,)


@register
def register_triangle_lower(op: TriangleLower, x: CUDATensor):
    a, b, c, d = 1, 1, 1, 1
    for dim in x.shape[:-2]:
        a *= dim
    b = x.shape[-2]
    c = x.shape[-1]
    assert b == c
    z = CUDATensor.allocate(x.shape, x.dtype)
    diagonal = op.diagonal
    kernel = generate_triangle_kernel("triangle_lower", f"index[1] > index[2] + {diagonal}")
    kernel(
        None,
        ctypes.c_void_p(x.data_ptr),
        ctypes.c_void_p(z.data_ptr),
        ctypes.c_size_t(a),
        ctypes.c_size_t(b),
        ctypes.c_size_t(c),
        ctypes.c_size_t(d),
    )
    return (z,)
