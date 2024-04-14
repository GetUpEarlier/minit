from typing import Union

from ..core.tensor import Tensor

from .tensor import TraceTensor
from ..graph import InternalNode, ConstantNode, OperatorNode
from ..core.operator import Operator
from ..core.dispatch import dispatch, register_dispatch
from ..core.object import match_pattern


def any_trace_tensor(*args):
    for arg in args:
        if match_pattern(TraceTensor, arg):
            return True
    return False


@register_dispatch(predicate=any_trace_tensor, priority=1)
def dispatch_any(op: Operator, *args: Union[TraceTensor, Tensor]):
    arg_values = []
    arg_nodes = []
    builder = None
    for arg in args:
        if isinstance(arg, TraceTensor):
            if builder is None:
                builder = arg._builder
            else:
                assert builder == arg._builder
    assert builder is not None
    for arg in args:
        if isinstance(arg, TraceTensor):
            arg_values.append(arg._value)
            arg_nodes.append(arg._node)
        else:
            arg_values.append(arg)
            arg_nodes.append(builder.create_constant(arg))
    output_values = dispatch(op, *arg_values)
    output_nodes = builder.create_operator(op, arg_nodes, len(output_values))
    return tuple(TraceTensor(builder, output_node, output_value) for output_node, output_value in zip(output_nodes, output_values))
