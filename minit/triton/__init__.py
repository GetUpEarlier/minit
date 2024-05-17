from ast import literal_eval
import contextlib
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union

from ..core.shape import to_immediate_shape
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
    dtype: str

@dataclass
class Store(Operator):
    ...

@dataclass
class Stage(Operator):
    ...

@register_dispatch(priority=-1)
def dispatch_load(op: Load, pointers: Tensor, mask: Tensor, other: Tensor):
    value = MetaTensor(pointers.shape, op.dtype)
    return (value,)

@register_dispatch(priority=-1)
def dispatch_store(_op: Store, pointers: Tensor, value: Tensor, mask: Tensor):
    return ()

@register_dispatch(priority=-1)
def dispatch_stage(_op: Stage, x: Tensor):
    return (x,)

def load(pointers: Tensor, mask: Tensor, other: Tensor, dtype: str):
    # from ..functional.arith import constant
    from ..functional.generate import fill
    if not isinstance(other, Tensor):
        other = fill(other, pointers.shape, dtype)
        # other = constant(other, pointers.dtype).add_axis(0, pointers.size).expand(0, pointers.shape)
    (values,) = dispatch(Load(dtype), pointers, mask, other)
    return values

def stage(value: Tensor):
    (value,) = dispatch(Stage(), value)
    return value

def store(pointers: Tensor, value: Tensor, mask: Tensor):
    () = dispatch(Store(), pointers, value, mask)

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


def generate_triton_kernel(graph: SubGraph, name: str, inputs: Sequence[str]):
    assert isinstance(graph, SubGraph)
    # assume first input is program id
    builder = ProgramBuilder()
    input_names = [f"_input_{name}" for name in inputs]
    input_signatures = [
        f"{input_name}: tl.{input_var.target.dtype}" for input_var, input_name in zip(graph.inputs[1:], input_names)
    ]
    builder.new_line("import triton")
    builder.new_line("import triton.language as tl")
    builder.new_line(f"def {name}({', '.join(input_signatures)}):")

    def generate_from_graph(graph: SubGraph, inputs: List[str]):
        local_variables: Dict[TensorNode, Variable] = {}

        def get_local(x: Use):
            if isinstance(x, ShapeUse):
                return Variable("constant", (), str(local_variables[x.target].shape[x.axis]), "int32")
            elif isinstance(x, ValueUse):
                if x.target in local_variables:
                    return local_variables[x.target]
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
            else:
                assert False, f"unknown local {x}"

        def define_local(x: TensorNode, value: Variable):
            assert isinstance(value, Variable)
            local_variables[x] = value

        def get_locals(*args: Use) -> Sequence[Variable]:
            results = []
            for arg in args:
                results.append(get_local(arg))
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
        
        def as_constant(x: Variable) -> Number:
            assert x.type == "constant"
            return dtype_info(x.dtype).python_type(literal_eval(x.value))
        
        def as_tensor(x: Variable) -> str:
            if x.type == "constant":
                return f"tl.full((), {dtype_info(x.dtype).python_type.__name__}('{x.value}'), tl.{x.dtype})"
            elif x.type == "scalar":
                return f"tl.full((), {x.value}, tl.{x.dtype})"
            elif x.type == "tensor":
                return x.value
            else:
                assert False

        for i, (input_name, input_var) in enumerate(zip(inputs, graph.inputs, strict=True)):
            assert input_var.axis is None
            if i == 0:
                define_local(input_var.target, Variable("tensor", (), input_name, input_var.target.dtype))
            else:
                define_local(input_var.target, Variable("scalar", (), input_name, input_var.target.dtype))

        for op_node in graph.operators:
            op = op_node.operator
            op_args = get_locals(*op_node.args)
            op_outputs = op_node.outputs
            if isinstance(op, operator.arith.Add):
                lhs, rhs = op_args
                compare_shapes(lhs, rhs)
                define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value}) + ({rhs.value})", lhs.dtype))
            elif isinstance(op, operator.arith.Subtract):
                lhs, rhs = op_args
                compare_shapes(lhs, rhs)
                define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value}) - ({rhs.value})", lhs.dtype))
            elif isinstance(op, operator.arith.Multiply):
                lhs, rhs = op_args
                compare_shapes(lhs, rhs)
                define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value}) * ({rhs.value})", lhs.dtype))
            elif isinstance(op, operator.arith.Divide):
                lhs, rhs = op_args
                compare_shapes(lhs, rhs)
                define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value}) / ({rhs.value})", lhs.dtype))
            elif isinstance(op, operator.control_flow.ForLoop):
                start, stop, step = op_args
                iterator = Variable("scalar", (), builder.new_local("i"), start.dtype)
                builder.new_line(f"for {iterator} in range({start.value}, {stop.value}, {step.value}):")
                with builder.indent():
                    generate_from_graph(op.body, [iterator])
            elif isinstance(op, Load):
                pointer, mask, other = op_args
                stage = Variable("tensor", pointer.shape, builder.new_local("load"), op.dtype)
                builder.new_line(f"{stage.value} = tl.load(({as_tensor(pointer)}).to(tl.pointer_type(tl.{op.dtype})), mask={as_tensor(mask)}, other={as_tensor(other)})")
                define_local(op_outputs[0], stage)
            elif isinstance(op, Store):
                pointer, value, mask = op_args
                builder.new_line(f"tl.store(({as_tensor(pointer)}).to(tl.pointer_type(tl.{value.dtype})), {as_tensor(value)}, mask={as_tensor(mask)})")
            elif isinstance(op, Stage):
                (x,) = op_args
                stage = Variable("tensor", x.shape, builder.new_local("stage"), x.dtype)
                builder.new_line(f"{stage.value} = {as_tensor(x)}")
                define_local(op_outputs[0], stage)
            elif isinstance(op, operator.arith.Cast):
                (x,) = op_args
                define_local(op_outputs[0], Variable("tensor", x.shape, f"{as_tensor(x)}.to(tl.{op.dtype})", op.dtype))
            elif isinstance(op, operator.arith.Exponential):
                (x,) = op_args
                define_local(op_outputs[0], Variable("tensor", x.shape, f"tl.exp({as_tensor(x)})", x.dtype))
            elif isinstance(op, operator.shape.AddAxis):
                (x,) = op_args
                define_local(op_outputs[0], Variable("tensor", x.shape[:op.axis]+(1,)+x.shape[op.axis:], f"tl.expand_dims({as_tensor(x)}, {op.axis})", x.dtype))
            elif isinstance(op, operator.shape.Broadcast):
                source, size = op_args
                source_shape = source.shape
                ndim = len(source_shape)
                target_shape = tuple([source_shape[i] for i in range(op.axis)] + [int(as_constant(size))] + [f"{source_shape[i]}" for i in range(op.axis+1, ndim)])
                target_shape_str = ",".join([f"{target_shape[i]}" for i in range(ndim)])
                define_local(op_outputs[0], Variable("tensor", target_shape, f"tl.broadcast_to({as_tensor(source)}, [{target_shape_str}])", source.dtype))
            elif isinstance(op, operator.arith.GreaterThan):
                lhs, rhs = op_args
                compare_shapes(lhs, rhs)
                define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value}) > ({rhs.value})", "bool"))
            elif isinstance(op, operator.arith.Not):
                (x,) = op_args
                define_local(op_outputs[0], Variable(x.type, lhs.shape, f"not ({x.value})", "bool"))
            elif isinstance(op, operator.arith.Equal):
                lhs, rhs = op_args
                compare_shapes(lhs, rhs)
                define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value}) == ({rhs.value})", "bool"))
            elif isinstance(op, operator.arith.And):
                lhs, rhs = op_args
                compare_shapes(lhs, rhs)
                define_local(op_outputs[0], Variable(type_promote(lhs, rhs), lhs.shape, f"({lhs.value}) and ({rhs.value})", "bool"))
            elif isinstance(op, operator.generate.GenerateInterval):
                start, stop, step = op_args
                as_constant(step) == 1
                shape = (int(as_constant(stop)) - int(as_constant(start)),)
                define_local(op_outputs[0], Variable("tensor", shape, f"tl.arange({start.value}, {stop.value})", start.dtype))
            elif isinstance(op, operator.generate.Fill):
                sizes = op_args
                shape = tuple(as_constant(size) for size in sizes)
                define_local(op_outputs[0], Variable("tensor", shape, f"tl.full({str(shape)}, {dtype_info(op.dtype).python_type.__name__}('{op.value}'), tl.{op.dtype})", op.dtype))
            elif isinstance(op, operator.arith.Constant):
                sizes = op_args
                shape = tuple(as_constant(size) for size in sizes)
                define_local(op_outputs[0], Variable("tensor", shape, f"tl.full({str(shape)}, {dtype_info(op.dtype).python_type.__name__}('{op.value}'), tl.{op.dtype})", op.dtype))
            elif isinstance(op, operator.reduce.Max):
                (x,) = op_args
                shape = x.shape[:op.axis] + (1,) + x.shape[op.axis+1:]
                define_local(op_outputs[0], Variable("tensor", shape, f"tl.expand_dims(tl.max({as_tensor(x)}, axis={op.axis}), {op.axis})", x.dtype))
            elif isinstance(op, operator.reduce.Sum):
                (x,) = op_args
                shape = x.shape[:op.axis] + (1,) + x.shape[op.axis+1:]
                define_local(op_outputs[0], Variable("tensor", shape, f"tl.expand_dims(tl.sum({as_tensor(x)}, axis={op.axis}), {op.axis})", x.dtype))
            else:
                assert False, f"unsupported operation {op}"

    with builder.indent():
        builder.new_line(f"_program_id = tl.program_id(0)")
        generate_from_graph(graph, ["_program_id", *input_names])

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
