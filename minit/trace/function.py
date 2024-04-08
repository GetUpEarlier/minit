from typing import Any, Callable, Protocol, Tuple

from .tensor import ConstantNode, InputNode, TraceGraph, TraceTensor

from ..core.tensor import Tensor


class TraceableFunction(Protocol):
    def __call__(self, *args: Tensor) -> Tuple[Tensor, ...]:
        ...


def trace_function(func: TraceableFunction, args: Tuple[Tensor, ...]) -> TraceGraph:
    graph = TraceGraph()
    inputs = tuple(TraceTensor(InputNode(graph, i), arg) for i, arg in enumerate(args))
    graph.inputs = tuple(inp._node for inp in inputs)
    outputs = func(*inputs)
    output_nodes = []
    for output in outputs:
        if not isinstance(output, TraceTensor):
            output = TraceTensor(ConstantNode(graph, output), output)
        output_nodes.append(output._node)
    graph.outputs = tuple(output_nodes)
    return graph
