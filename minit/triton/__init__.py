import contextlib
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..functional.utils import _broadcast_constant
from ..core.meta import MetaTensor
from ..core.dispatch import dispatch, register_dispatch
from ..core.tensor import Tensor
from ..core.operator import Operator
from ..core.scalar import ScalarTensor
from ..graph import ConstantNode, SubGraph, TensorNode
from .. import operator

@dataclass
class Load(Operator):
    dtype: str

@dataclass
class Store(Operator):
    ...

@register_dispatch(priority=-1)
def dispatch_load(op: Load, pointers: Tensor, mask: Tensor, other: Tensor):
    value = MetaTensor(pointers.shape, op.dtype)
    return (value,)

@register_dispatch(priority=-1)
def dispatch_store(_op: Store, pointers: Tensor, value: Tensor, mask: Tensor):
    return (pointers,)

def load(pointers: Tensor, mask: Tensor, other: Tensor, dtype: str):
    from ..functional.arith import constant
    if not isinstance(other, Tensor):
        other = constant(other, pointers.dtype).broadcast(0, pointers.size).expand(0, pointers.shape)
    (values,) = dispatch(Load(dtype), pointers, mask, other)
    return values

def store(pointers: Tensor, value: Tensor, mask: Tensor):
    (pointers,) = dispatch(Store(), pointers, value, mask)
    return pointers

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


def generate_triton_kernel(graph: SubGraph, name: str, inputs: Sequence[str]):
    # assume first input is program id
    builder = ProgramBuilder()
    input_names = [f"_input_{name}" for name in inputs]
    builder.new_line("import triton")
    builder.new_line("import triton.language")
    builder.new_line(f"def {name}({','.join(input_names)}):")

    def make_constant(x: ConstantNode):
        assert isinstance(x.value, ScalarTensor)
        if len(x.value.shape) == 0:
            return f"({x.value._value})"
        else:
            return f"triton.language.full({x.value._shape}, {x.value._value}, triton.language.{x.value._dtype})"

    def generate_from_graph(graph: SubGraph, inputs: List[str]):
        kernel_locals = {input_var(): input_name for input_name, input_var in zip(inputs, graph.inputs, strict=True)}

        def get_local(x: TensorNode):
            if isinstance(x, ShapeNode):
                return get_local(x.source.value())
            elif isinstance(x, ConstantNode):
                return make_constant(x)
            else:
                return kernel_locals[x]

        def define_local(x: TensorNode, expression: str):
            assert x not in kernel_locals
            name = builder.new_local("")
            builder.new_line(f"{name} = {expression}")
            kernel_locals[x] = name

        def bind_local(x: TensorNode, expression: str):
            kernel_locals[x] = expression

        def define_temp(expression: str):
            name = builder.new_local("")
            builder.new_line(f"{name} = {expression}")
            return name

        for op_ref in graph.operators:
            op_node = op_ref()
            op = op_node.operator
            if isinstance(op, operator.arith.Add):
                lhs, rhs = op_node.args
                define_local(op_node.outputs[0].value(), f"{get_local(lhs())} + {get_local(rhs())}")
            elif isinstance(op, operator.arith.Subtract):
                lhs, rhs = op_node.args
                define_local(op_node.outputs[0].value(), f"{get_local(lhs())} - {get_local(rhs())}")
            elif isinstance(op, operator.arith.Multiply):
                lhs, rhs = op_node.args
                define_local(op_node.outputs[0].value(), f"{get_local(lhs())} * {get_local(rhs())}")
            elif isinstance(op, operator.arith.Divide):
                lhs, rhs = op_node.args
                define_local(op_node.outputs[0].value(), f"{get_local(lhs())} / {get_local(rhs())}")
            elif isinstance(op, operator.control_flow.ForLoop):
                start, stop, step, *args = op_node.args
                iterator = builder.new_local("i")
                builder.new_line(f"for {iterator} in range({get_local(start())}, {get_local(stop())}, {get_local(step())}):")
                arg_names = [kernel_locals[arg()] for arg in args]
                with builder.indent():
                    generate_from_graph(op.body, [iterator, *arg_names])
            elif isinstance(op, Load):
                pointer, mask, other = op_node.args
                define_local(op_node.outputs[0].value(), f"triton.language.load({get_local(pointer())}, mask={get_local(mask())}, other={get_local(other())})")
            elif isinstance(op, Store):
                pointer, value, mask = op_node.args
                builder.new_line(f"triton.language.store({get_local(pointer())}, {get_local(value())}, mask={get_local(mask())})")
                define_local(op_node.outputs[0].value(), kernel_locals[pointer()])
            elif isinstance(op, operator.arith.Cast):
                define_local(op_node.outputs[0].value(), f"{get_local(op_node.args[0]())}.to(triton.language.{op.dtype})")
            elif isinstance(op, operator.shape.AddAxis):
                define_local(op_node.outputs[0].value(), f"triton.language.expand_dims({get_local(op_node.args[0]())}, {op.axis})")
            elif isinstance(op, operator.shape.Broadcast):
                source_shape = op_node.args[0]().shape
                size = get_local(op_node.args[1]())
                ndim = len(op_node.args[0]().shape)
                target_shape = ",".join([f"{source_shape[i]._value}" for i in range(op.axis)] + [size] + [f"{source_shape[i]._value}" for i in range(op.axis+1, ndim)])
                define_local(op_node.outputs[0].value(), f"triton.language.broadcast_to({get_local(op_node.args[0]())}, ({target_shape},))")
            elif isinstance(op, operator.arith.GreaterThan):
                lhs, rhs = op_node.args
                define_local(op_node.outputs[0].value(), f"{get_local(lhs())} > {get_local(rhs())}")
            elif isinstance(op, operator.arith.Not):
                [x] = op_node.args
                define_local(op_node.outputs[0].value(), f"(not {get_local(x())})")
            elif isinstance(op, operator.arith.Equal):
                lhs, rhs = op_node.args
                define_local(op_node.outputs[0].value(), f"{get_local(lhs())} == {get_local(rhs())}")
            elif isinstance(op, operator.arith.And):
                lhs, rhs = op_node.args
                define_local(op_node.outputs[0].value(), f"{get_local(lhs())} and {get_local(rhs())}")
            elif isinstance(op, operator.generate.GenerateInterval):
                start, stop, step = op_node.args
                assert step().value.item() == 1
                define_local(op_node.outputs[0].value(), f"triton.language.arange({get_local(start())}, {get_local(stop())})")
            else:
                assert False, f"unsupported operation {op}"

    with builder.indent():
        builder.new_line(f"_program_id = triton.language.program_id(0)")
        generate_from_graph(graph, ["_program_id", *input_names])

    return builder.build()
