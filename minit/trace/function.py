from typing import Optional, Protocol, Sequence, Tuple

from ..core.meta import MetaTensor
from ..graph import GraphBuilder, SubGraph, Graph, Use
from .tensor import TraceTensor
from ..core.tensor import Tensor


class TraceableFunction(Protocol):
    def __call__(self, *args: Tensor) -> Optional[Tuple[Tensor, ...]]:
        ...


def trace_function(func: TraceableFunction, args: Sequence[Tensor]) -> SubGraph:
    graph = Graph()
    input_nodes = tuple(graph.create_input(MetaTensor(arg.shape, arg.dtype)) for arg in args)
    builder = GraphBuilder(graph, input_nodes, graph.operators.view())
    output_nodes = trace_function_on_graph(func, args, builder, [input_node.use_value(None) for input_node in input_nodes])
    if output_nodes is None:
        output_nodes = ()
    return SubGraph(graph, tuple(input_node.use_value(None) for input_node in input_nodes), graph.operators.view(), tuple(output_nodes))


def trace_function_on_graph(func: TraceableFunction, args: Tuple[Tensor, ...], builder: GraphBuilder, uses: Sequence[Use]) -> Tuple[Use, ...]:
    inputs = tuple(TraceTensor(builder, use, arg) for i, (arg, use) in enumerate(zip(args, uses)))
    outputs = func(*inputs)
    if outputs is None:
        outputs = ()
    output_nodes = []
    for output in outputs:
        if not isinstance(output, TraceTensor):
            output = TraceTensor(builder, builder.create_constant(output), output)
        output_nodes.append(output._node)
    return builder.build(*output_nodes)
