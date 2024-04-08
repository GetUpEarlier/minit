from collections import defaultdict
from typing import Dict

from ..core.dispatch import dispatch

from ..core.tensor import Tensor
from .tensor import ConstantNode, TraceGraph, TraceNode


class TraceGraphExecutor:
    graph: TraceGraph
    reference_counts: Dict[TraceNode, int]

    def __init__(self, graph: TraceGraph) -> None:
        self.graph = graph
        counts = defaultdict(lambda: 0)
        for operator in graph.operators:
            for arg in operator.args:
                counts[arg] += 1
        for output in graph.outputs:
            counts[output] += 1
        self.reference_counts = counts

    def execute(self, *args: Tensor):
        assert len(args) == len(self.graph.inputs)
        values = {
            node: value for node, value in zip(self.graph.inputs, args) if self.reference_counts.get(node, 0) != 0
        }
        reference_counts = {}

        def consume_value(node: TraceNode):
            if isinstance(node, ConstantNode):
                return node.value
            value = values[node]
            reference_counts[node] -= 1
            if reference_counts[node] == 0:
                del reference_counts[node]
                del values[node]
            return value
        
        def produce_value(node: TraceNode, value: Tensor):
            values[node] = value
            reference_counts[node] = self.reference_counts[node]

        for node, value in zip(self.graph.inputs, args):
            produce_value(node, value)

        for operator in self.graph.operators:
            args = tuple(map(consume_value, operator.args))
            outputs = dispatch(operator.operator, *args)
            for node, output in zip(operator.outputs, outputs):
                produce_value(node, output)

        outputs = tuple(map(consume_value, self.graph.outputs))
        return outputs
