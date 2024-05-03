from typing import Any, Callable, Protocol, Sequence, Tuple

from ..core.meta import MetaTensor

from ..graph import GraphBuilder, SubGraph, Graph, TensorNodeRef

from .tensor import TraceTensor

from ..core.tensor import Tensor


class TraceableFunction(Protocol):
    def __call__(self, *args: Tensor) -> Tuple[Tensor, ...]:
        ...


def trace_function(func: TraceableFunction, args: Sequence[Tensor]) -> SubGraph:
    builder = GraphBuilder(Graph())
    inputs = tuple(TraceTensor(builder, builder.create_input(MetaTensor(arg.shape, arg.dtype)), arg) for i, arg in enumerate(args))
    outputs = func(*inputs)
    output_nodes = []
    for output in outputs:
        if not isinstance(output, TraceTensor):
            output = TraceTensor(builder, builder.create_constant(output), output)
        output_nodes.append(output._node)
    graph = builder.build(*output_nodes)
    return graph


def trace_function_on_graph(func: TraceableFunction, args: Tuple[Tensor, ...], graph: Graph, nodes: Tuple[TensorNodeRef, ...]) -> SubGraph:
    builder = GraphBuilder(graph)
    inputs = tuple(TraceTensor(builder, node, arg) for i, (arg, node) in enumerate(zip(args, nodes)))
    outputs = func(*inputs)
    output_nodes = []
    for output in outputs:
        if not isinstance(output, TraceTensor):
            output = TraceTensor(builder, builder.create_constant(output), output)
        output_nodes.append(output._node)
    result_graph = builder.build(*output_nodes)
    result_graph.inputs = list(nodes)
    return result_graph
