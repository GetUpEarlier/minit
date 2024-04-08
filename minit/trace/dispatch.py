from typing import Union

from ..core.tensor import Tensor

from .tensor import InternalNode, ConstantNode, TraceOperatorNode, TraceTensor
from ..core.operator import Operator
from ..core.dispatch import dispatch, register_dispatch
from ..core.object import match_pattern


def any_trace_tensor(*args):
    for arg in args:
        if match_pattern(TraceTensor, arg):
            return True
    return False


@register_dispatch(predicate=any_trace_tensor)
def dispatch_any(op: Operator, *args: Union[TraceTensor, Tensor]):
    arg_values = []
    arg_nodes = []
    graph = None
    for arg in args:
        if isinstance(arg, TraceTensor):
            if graph is None:
                graph = arg._node.graph
            else:
                assert graph == arg._node.graph
    assert graph is not None
    for arg in args:
        if isinstance(arg, TraceTensor):
            arg_values.append(arg._value)
            arg_nodes.append(arg._node)
        else:
            arg_values.append(arg)
            arg_nodes.append(ConstantNode(graph, arg))
    output_values = dispatch(op, *arg_values)
    operator_node = TraceOperatorNode(op, tuple(arg_nodes))
    output_nodes = tuple(InternalNode(graph, operator_node, i) for i in range(len(output_values)))
    operator_node.outputs = output_nodes
    graph.operators.append(operator_node)
    return tuple(TraceTensor(output_node, output_value) for output_node, output_value in zip(output_nodes, output_values))
