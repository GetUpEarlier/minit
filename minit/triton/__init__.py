from ast import literal_eval
import contextlib
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union

from ..core.shape import to_immediate_shape, to_symbolic_shape
from ..trace.function import trace_function
from ..compiler.python import pythonc
from ..core.dtype import dtype_info
from ..functional.utils import _broadcast_constant
from ..core.meta import MetaTensor
from ..core.dispatch import dispatch, register_dispatch
from ..core.tensor import Tensor
from ..core.operator import Operator
from ..core.scalar import ScalarTensor
from ..graph import ConstantNode, ShapeUse, SubGraph, TensorNode, Use, ValueUse
from .. import operator

@dataclass
class Load(Operator):
    name: str
    dtype: str

@dataclass
class LoadBlock(Operator):
    name: str
    dtype: str
    ndim: int

@dataclass
class StoreBlock(Operator):
    ndim: int

@dataclass
class Store(Operator):
    ...

@dataclass
class AtomicAdd(Operator):
    ...

@dataclass
class Stage(Operator):
    name: str

@dataclass
class MakePointer(Operator):
    dtype: str

@register_dispatch(priority=-1)
def dispatch_load(op: Load, pointers: Tensor, mask: Tensor, other: Tensor):
    value = MetaTensor(pointers.shape, op.dtype)
    return (value,)

@register_dispatch(priority=-1)
def dispatch_load_block(op: LoadBlock, base: Tensor, *args: Tensor):
    nargs = len(args)
    ndim = nargs // 4
    assert ndim * 4 == nargs
    block_shape = args[3*ndim:4*ndim]
    value = MetaTensor(block_shape, op.dtype)
    return (value,)

@register_dispatch(priority=-1)
def dispatch_store(_op: Store, value: Tensor, pointers: Tensor, mask: Tensor):
    return ()

@register_dispatch(priority=-1)
def dispatch_atomic_add(_op: AtomicAdd, value: Tensor, pointers: Tensor, mask: Tensor):
    return ()

@register_dispatch(priority=-1)
def dispatch_store_block(op: StoreBlock, value: Tensor, base: Tensor, *args: Tensor):
    return ()

@register_dispatch(priority=-1)
def dispatch_stage(_op: Stage, x: Tensor):
    return (x,)

@register_dispatch(priority=-1)
def dispatch_stage(_op: MakePointer, base: Tensor, offset: Tensor):
    pointer = MetaTensor(offset.shape, "int64")
    return (pointer,)

def load(name: str, pointers: Tensor, mask: Tensor, other: Tensor, dtype: str) -> Tensor:
    # from ..functional.arith import constant
    from ..functional.generate import fill
    if not isinstance(other, Tensor):
        other = fill(other, pointers.shape, dtype)
        # other = constant(other, pointers.dtype).add_axis(0, pointers.size).expand(0, pointers.shape)
    (value,) = dispatch(Load(name, dtype), pointers, mask, other)
    return value

def make_pointer(base: Tensor, offset: Tensor, dtype: str) -> Tensor:
    assert len(base.shape) == 0
    (pointer,) = dispatch(MakePointer(dtype), base, offset.cast("int64"))
    return pointer

def load_block(
        name: str,
        base: Tensor,
        shape: Sequence[Union[Tensor, Number]],
        strides: Optional[Sequence[Tensor]],
        offsets: Sequence[Optional[Tensor]],
        block_shape: Sequence[Optional[Union[Tensor, Number]]],
        dtype: str
) -> Tensor:
    from ..functional.arith import constant
    from ..functional.utils import _convert_constant
    shape = _convert_constant(*shape, dtype="int64")
    ndim = len(shape)
    if strides is None:
        strides = []
        size = constant(1, "int64")
        for dim in reversed(shape):
            strides.append(size)
            size = size * dim
        strides.reverse()
    assert len(strides) == ndim
    strides = _convert_constant(*strides, dtype="int64")
    offsets = list(offsets)
    assert len(offsets) == ndim
    for i in range(len(offsets)):
        if offsets[i] is None:
            offsets[i] = constant(0, "int32")
        else:
            offsets[i] = offsets[i].cast("int32")
    assert len(block_shape) == ndim
    block_shape = list(block_shape)
    remove_axes = []
    for i in range(len(block_shape)):
        if block_shape[i] is None:
            block_shape[i] = 1
            remove_axes.append(i)
    block_shape = _convert_constant(*block_shape, dtype="int64")
    (values,) = dispatch(LoadBlock(name, dtype, ndim), base, *shape, *strides, *offsets, *block_shape)
    for axis in reversed(remove_axes):
        values = values.remove_axis(axis)
    assert base.dtype == "int64"
    return values

def stage(name: str, value: Tensor):
    (value,) = dispatch(Stage(name), value)
    return value

def store(value: Tensor, pointers: Tensor, mask: Tensor):
    () = dispatch(Store(), pointers, value, mask)

def atomic_add(value: Tensor, pointers: Tensor, mask: Tensor):
    () = dispatch(AtomicAdd(), pointers, value, mask)

def store_block(
        value: Tensor,
        base: Tensor,
        shape: Sequence[Union[Tensor, Number]],
        strides: Optional[Sequence[Tensor]],
        offsets: Sequence[Optional[Tensor]],
        block_shape: Sequence[Optional[Union[Tensor, Number]]],
        dtype: str
):
    from ..functional.arith import constant
    from ..functional.utils import _convert_constant
    shape = _convert_constant(*shape, dtype="int64")
    ndim = len(shape)
    if strides is None:
        strides = []
        size = constant(1, "int64")
        for dim in reversed(shape):
            strides.append(size)
            size = size * dim
    assert len(strides) == ndim
    strides = _convert_constant(*strides, dtype="int64")
    offsets = list(offsets)
    assert len(offsets) == ndim
    for i in range(len(offsets)):
        if offsets[i] is None:
            offsets[i] = constant(0, "int32")
        else:
            offsets[i] = offsets[i].cast("int32")
    assert len(block_shape) == ndim
    block_shape = list(block_shape)
    add_axes = []
    for i in range(len(block_shape)):
        if block_shape[i] is None:
            block_shape[i] = 1
            add_axes.append(i)
    block_shape = _convert_constant(*block_shape, dtype="int64")
    for axis in add_axes:
        value = value.add_axis(axis, None)
    assert base.dtype == "int64"
    () = dispatch(StoreBlock(ndim), value, base, *shape, *strides, *offsets, *block_shape)

class ProgramBuilder:
    locals: List[str]
    lines: List[str]
    indent_count: int

    def __init__(self) -> None:
        self.locals = []
        self.lines = []
        self.indent_count = 0

    def new_local(self, name: str) -> str:
        self.locals.append(f"_local_{len(self.locals)}_{name}")
        return self.locals[-1]

    @contextlib.contextmanager
    def indent(self):
        self.indent_count += 4
        yield
        self.indent_count -= 4

    def new_line(self, statement: str):
        self.lines.append(" " * self.indent_count + statement)

    def build(self):
        content = ""
        for line in self.lines:
            content += line
            content += "\n"
        return content
    

VariableType = Union[Literal["constant"], Literal["scalar"], Literal["tensor"]]


@dataclass
class Variable:
    type: VariableType
    shape: Tuple[int, ...]
    value: str
    dtype: str

    def __post_init__(self):
        for dim in self.shape:
            assert isinstance(dim, int)
        assert isinstance(self.value, str)



def generate_from_graph(builder: ProgramBuilder, graph: SubGraph, inputs: List[Tuple[str, Optional[Tuple[int, ...]]]]) -> List[Variable]:
    local_variables: Dict[TensorNode, Variable] = {}
    def get_local(x: Use):
        if isinstance(x, ShapeUse):
            return Variable("constant", (), str(local_variables[x.target].shape[x.axis]), "int32")
        elif isinstance(x, ValueUse):
            if x.target in local_variables:
                result = local_variables[x.target]
                assert result is not None
                return result
            if isinstance(x.target, ConstantNode):
                if len(x.target.value.shape) == 0:
                    return Variable("constant", (), str(x.target.value.item()), x.target.dtype)
                else:
                    assert isinstance(x.target.value, ScalarTensor)
                    shape = to_immediate_shape(x.target.value.shape)
                    dtype = x.target.dtype
                    item = x.target.value.value()
                    return Variable("tensor", shape,
                                    f"tl.full({shape}, {dtype_info(dtype).python_type.__name__}('{item}'), tl.{dtype})", dtype)
            assert False, "unknown node from outside"
        else:
            assert False, f"unknown local {x}"
    def define_local(x: TensorNode, value: Variable):
        assert isinstance(value, Variable)
        assert len(value.shape) == len(x.shape), f"{len(value.shape)} vs {len(x.shape)}"
        local_variables[x] = value
    def stage_local(name: str, value: Variable):
        variable_name = builder.new_local(name)
        builder.new_line(f"{variable_name} = {value.value} # {value.shape} {value.dtype}")
        variable = Variable(value.type, value.shape, variable_name, value.dtype)
        return variable
    def get_locals(*args: Use) -> Sequence[Variable]:
        results = []
        for arg in args:
            result = get_local(arg)
            assert isinstance(result, Variable)
            results.append(result)
        return tuple(results)
    def type_promote(*vars: Variable):
        assert len(vars) > 0
        result = "constant"
        for var in vars:
            if var.type == "constant":
                pass
            elif var.type == "scalar":
                if result == "constant":
                    result = "scalar"
            elif var.type == "tensor":
                result = "tensor"
        return result
    def compare_shapes(*vars: Variable) -> bool:
        assert len(vars) > 0
        shape = vars[0].shape
        for var in vars[1:]:
            if var.shape != shape:
                return False
        return True
    def as_constant_repr(x: Variable) -> str:
        assert x.type == "constant"
        return f"{dtype_info(x.dtype).python_type.__name__}('{x.value}')"
    def as_constant(x: Variable) -> Number:
        return eval(as_constant_repr(x))
    def as_tensor(x: Variable) -> str:
        if x.type == "constant":
            return f"tl.full((), {as_constant_repr(x)}, tl.{x.dtype})"
        elif x.type == "scalar":
            return f"tl.full((), {x.value}, tl.{x.dtype})"
        elif x.type == "tensor":
            return x.value
        else:
            assert False
    for i, ((input_name, input_shape), input_var) in enumerate(zip(inputs, graph.inputs, strict=True)):
        assert input_var.axis is None
        if input_shape is None:
            input_shape = [dim.item() for dim in input_var.target.meta.shape]
        else:
            assert len(input_var.target.meta.shape) == len(input_shape)
        define_local(input_var.target, Variable("tensor", tuple(input_shape), input_name, input_var.target.dtype))
    for op_node in graph.operators:
        op = op_node.operator
        op_args = get_locals(*op_node.args)
        op_outputs = op_node.outputs
        if isinstance(op, operator.arith.Add):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} + {rhs.value})", lhs.dtype))
        elif isinstance(op, operator.arith.Subtract):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} - {rhs.value})", lhs.dtype))
        elif isinstance(op, operator.arith.Multiply):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} * {rhs.value})", lhs.dtype))
        elif isinstance(op, operator.arith.Divide):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} / {rhs.value})", lhs.dtype))
        elif isinstance(op, operator.arith.FloorDivide):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} // {rhs.value})", lhs.dtype))
        elif isinstance(op, operator.arith.Modulo):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} % {rhs.value})", lhs.dtype))
        elif isinstance(op, operator.control_flow.ForLoop):
            count, *variables = op_args
            iterator = Variable("scalar", (), builder.new_local("i"), count.dtype)
            loop_variables: List[Variable] = []
            num_loop_variables = len(op.body.outputs)
            num_constants = len(op.body.inputs) - num_loop_variables
            assert num_constants >= 0
            for i, variable in enumerate(variables):
                assert variable.type != "constant"
                if i < num_loop_variables:
                    loop_variable = stage_local("loop", variable)
                    loop_variables.append(loop_variable)
                else:
                    loop_variables.append(variable)
            builder.new_line(f"for {iterator.value} in range({count.value}):")
            with builder.indent():
                output_variables = generate_from_graph(
                    builder, op.body, [(iterator.value, iterator.shape), *((loop_variable.value, loop_variable.shape) for loop_variable in loop_variables)]
                )
                assert len(output_variables) == num_loop_variables
                for output, output_variable, loop_variable in zip(op_outputs, output_variables, loop_variables[:num_loop_variables], strict=True):
                    if loop_variable.value != output_variable.value:
                        builder.new_line(f"{loop_variable.value} = {output_variable.value}")
                    define_local(output, loop_variable)
        elif isinstance(op, Load):
            pointer, mask, other = op_args
            stage = Variable("tensor", pointer.shape, f"tl.load({as_tensor(pointer)}.to(tl.pointer_type(tl.{op.dtype})), mask={as_tensor(mask)}, other={as_tensor(other)})", op.dtype)
            define_local(op_outputs[0], stage_local(op.name, stage))
        elif isinstance(op, LoadBlock):
            base, *args = op_args
            nargs = len(args)
            ndim = nargs // 4
            shape = "[" + ",".join(dim.value for dim in args[:ndim]) + "]"
            strides = "[" + ",".join(dim.value for dim in args[ndim:2*ndim]) + "]"
            offsets = "[" + ",".join(dim.value for dim in args[2*ndim:3*ndim]) + "]"
            block_shape = tuple(as_constant(dim) for dim in args[3*ndim:4*ndim])
            order = tuple(reversed(range(ndim)))
            # order = tuple(range(ndim))
            stage = Variable("tensor", block_shape, f"tl.load(tl.make_block_ptr(base={as_tensor(base)}.to(tl.pointer_type(tl.{op.dtype})), shape={shape}, strides={strides}, offsets={offsets}, block_shape={block_shape}, order={order}))", op.dtype)
            define_local(op_outputs[0], stage_local(op.name, stage))
        elif isinstance(op, Store):
            pointer, value, mask = op_args
            builder.new_line(f"tl.store({as_tensor(pointer)}.to(tl.pointer_type(tl.{value.dtype})), {as_tensor(value)}, mask={as_tensor(mask)})")
        elif isinstance(op, AtomicAdd):
            pointer, value, mask = op_args
            builder.new_line(f"tl.atomic_add({as_tensor(pointer)}.to(tl.pointer_type(tl.{value.dtype})), {as_tensor(value)}, mask={as_tensor(mask)})")
        elif isinstance(op, StoreBlock):
            value, base, *args = op_args
            nargs = len(args)
            ndim = nargs // 4
            shape = "[" + ",".join(dim.value for dim in args[:ndim]) + "]"
            strides = "[" + ",".join(dim.value for dim in args[ndim:2*ndim]) + "]"
            offsets = "[" + ",".join(dim.value for dim in args[2*ndim:3*ndim]) + "]"
            block_shape = tuple(as_constant(dim) for dim in args[3*ndim:4*ndim])
            order = tuple(reversed(range(ndim)))
            builder.new_line(f"tl.store(tl.make_block_ptr(base={as_tensor(base)}.to(tl.pointer_type(tl.{value.dtype})), shape={shape}, strides={strides}, offsets={offsets}, block_shape={block_shape}, order={order}), {as_tensor(value)})")
        elif isinstance(op, Stage):
            (x,) = op_args
            stage = Variable("tensor", x.shape, f"{as_tensor(x)}", x.dtype)
            define_local(op_outputs[0], stage_local(op.name, stage))
        elif isinstance(op, operator.arith.Cast):
            (x,) = op_args
            define_local(op_outputs[0], Variable("tensor", x.shape, f"{as_tensor(x)}.to(tl.{op.dtype})", op.dtype))
        elif isinstance(op, operator.arith.Exponential):
            (x,) = op_args
            define_local(op_outputs[0], Variable("tensor", x.shape, f"tl.exp({as_tensor(x)})", x.dtype))
        elif isinstance(op, operator.shape.AddAxis):
            (x,) = op_args
            define_local(op_outputs[0], Variable("tensor", x.shape[:op.axis]+(1,)+x.shape[op.axis:], f"tl.expand_dims({as_tensor(x)}, {op.axis})", x.dtype))
        elif isinstance(op, operator.shape.RemoveAxis):
            (x,) = op_args
            shape = x.shape
            ndim = len(shape)
            assert shape[op.axis] == 1
            shape = shape[:op.axis] + shape[op.axis+1:]
            define_local(op_outputs[0], Variable("tensor", shape, f"tl.reshape({as_tensor(x)}, {shape})", x.dtype))
        elif isinstance(op, operator.shape.Broadcast):
            source, size = op_args
            source_shape = source.shape
            ndim = len(source_shape)
            target_shape = tuple([source_shape[i] for i in range(op.axis)] + [int(as_constant(size))] + [source_shape[i] for i in range(op.axis+1, ndim)])
            target_shape_str = ",".join([f"{target_shape[i]}" for i in range(ndim)])
            define_local(op_outputs[0], Variable("tensor", target_shape, f"tl.broadcast_to({as_tensor(source)}, [{target_shape_str}])", source.dtype))
        elif isinstance(op, operator.arith.GreaterThan):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} > {rhs.value})", "bool"))
        elif isinstance(op, operator.arith.Not):
            (x,) = op_args
            define_local(op_outputs[0], Variable(x.type, lhs.shape, f"(not {x.value})", "bool"))
        elif isinstance(op, operator.arith.Equal):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} == {rhs.value})", "bool"))
        elif isinstance(op, operator.arith.And):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value} and {rhs.value})", "bool"))
        elif isinstance(op, operator.generate.GenerateInterval):
            start, stop, step = op_args
            as_constant(step) == 1
            shape = (int(as_constant(stop)) - int(as_constant(start)),)
            define_local(op_outputs[0], Variable("tensor", shape, f"tl.arange({start.value}, {stop.value})", start.dtype))
        elif isinstance(op, operator.generate.Fill):
            value, *sizes = op_args
            shape = tuple(as_constant(size) for size in sizes)
            if value.type == "constant":
                define_local(op_outputs[0], Variable("tensor", shape, f"tl.full({str(shape)}, {as_constant_repr(value)}, tl.{value.dtype})", value.dtype))
            else:
                define_local(op_outputs[0], Variable("tensor", shape, f"tl.full({str(shape)}, {value.value}, tl.{value.dtype})", value.dtype))
        elif isinstance(op, operator.arith.Constant):
            sizes = op_args
            shape = tuple(as_constant(size) for size in sizes)
            define_local(op_outputs[0], Variable("constant", shape, str(op.value), op.dtype))
        elif isinstance(op, operator.reduce.Max):
            (x,) = op_args
            shape = x.shape[:op.axis] + (1,) + x.shape[op.axis+1:]
            define_local(op_outputs[0], Variable("tensor", shape, f"tl.max({as_tensor(x)}, axis={op.axis}, keep_dims=True)", x.dtype))
        elif isinstance(op, operator.reduce.Sum):
            (x,) = op_args
            shape = x.shape[:op.axis] + (1,) + x.shape[op.axis+1:]
            define_local(op_outputs[0], Variable("tensor", shape, f"tl.sum({as_tensor(x)}, axis={op.axis}, keep_dims=True)", x.dtype))
        elif isinstance(op, operator.shape.Transpose):
            (x,) = op_args
            shape = x.shape
            assert len(shape) == 2
            assert op.axis_a == 0
            assert op.axis_b == 1
            shape = shape[1], shape[0]
            define_local(op_outputs[0], Variable("tensor", shape, f"tl.trans({as_tensor(x)})", x.dtype))
            # (x,) = op_args
            # shape = list(x.shape)
            # ndim = len(shape)
            # axes = list(range(ndim))
            # axes[op.axis_a], axes[op.axis_b] = axes[op.axis_b], axes[op.axis_a]
            # shape[op.axis_a], shape[op.axis_b] = shape[op.axis_b], shape[op.axis_a]
            # define_local(op_outputs[0], Variable("tensor", tuple(shape), f"({as_tensor(x)}).permute({axes})", x.dtype))
        elif isinstance(op, operator.shape.Fold):
            (x,) = op_args
            size = 1
            for dim in x.shape[op.start:op.stop]:
                size *= dim
            shape = [*x.shape[:op.start], size, *x.shape[op.stop:]]
            define_local(op_outputs[0], Variable("tensor", tuple(shape), f"tl.reshape({as_tensor(x)}, {shape})", x.dtype))
        elif isinstance(op, operator.linalg.MatrixMultiply):
            (x, y) = op_args
            m, k0 = x.shape
            n, k1 = x.shape
            shape = [m, n]
            assert k0 == k1
            define_local(op_outputs[0], Variable("tensor", tuple(shape), f"tl.dot({as_tensor(x)}, tl.trans({as_tensor(y)}))", x.dtype))
        elif isinstance(op, operator.arith.SelectMax):
            lhs, rhs = op_args
            compare_shapes(lhs, rhs)
            define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"tl.maximum({lhs.value}, {rhs.value})", lhs.dtype))
        elif isinstance(op, operator.generate.GenerateSequence):
            start, size, step = op_args
            assert as_constant(step) == 1
            define_local(op_outputs[0], Variable("tensor", (as_constant(size),), f"(tl.arange(0, {as_constant(size)}) + {start.value})", start.dtype))
        elif isinstance(op, operator.arith.Select):
            condition, false_value, true_value = op_args
            assert condition.dtype == "bool"
            assert condition.shape == false_value.shape
            assert condition.shape == true_value.shape
            assert true_value.dtype == false_value.dtype
            define_local(op_outputs[0], Variable("tensor", condition.shape, f"tl.where({as_tensor(condition)}, {as_tensor(true_value)}, {as_tensor(false_value)})", true_value.dtype))
        elif isinstance(op, operator.arith.Logarithm):
            (x,) = op_args
            define_local(op_outputs[0], Variable("tensor", x.shape, f"tl.log({as_tensor(x)})", x.dtype))
        elif isinstance(op, MakePointer):
            (base, offset) = op_args
            assert base.dtype == "int64"
            assert offset.dtype == "int64"
            define_local(op_outputs[0], Variable("tensor", offset.shape, f"({as_tensor(base)}.to(tl.pointer_type(tl.{op.dtype})) + {offset.value}).to(tl.int64)", "int64"))
        else:
            assert False, f"unsupported operation {op}"
    output_variables = [get_local(output) for output in graph.outputs]
    return output_variables


def generate_triton_kernel(graph: SubGraph, name: str, inputs: Sequence[str]):
    assert isinstance(graph, SubGraph)
    # assume first input is program id
    builder = ProgramBuilder()
    input_names = [f"_input_{name}" for name in inputs]
    input_signatures = [
        f"{input_name}: tl.tensor" for input_var, input_name in zip(graph.inputs[1:], input_names)
    ]
    builder.new_line("import triton")
    builder.new_line("import triton.language as tl")
    builder.new_line(f"def {name}({', '.join(input_signatures)}):")

    with builder.indent():
        builder.new_line(f"_program_id = tl.program_id(0)")
        generate_from_graph(builder, graph, [("_program_id", None), *[(input_name, None) for input_name in input_names]])

    return builder.build()


class TraceableTritonFunction(Protocol):
    def __call__(self, *args: Tensor) -> None:
        ...


def trace_triton_jit(name: str, inputs: Dict[str, str]):
    def decorator(function: TraceableTritonFunction):
        args = []
        for input_dtype in ["int32", *inputs.values()]:
            args.append(MetaTensor((), input_dtype))
        kernel_graph = trace_function(function, args)
        kernel_name = name
        kernel = generate_triton_kernel(kernel_graph, name, list(inputs.keys()))
        print("Kernel:")
        print(kernel)
        kernel_function = pythonc.compile(kernel)[kernel_name]
        import triton
        triton_kernel = triton.jit(kernel_function)
        return triton_kernel
    return decorator
