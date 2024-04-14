from collections import defaultdict
from typing import Dict, Tuple

from ..core.dispatch import dispatch

from ..core.tensor import Tensor
from ..graph import ConstantNode, Graph, NodeBase, NodeRef, ShapeNode, SubGraph, TensorNode


class TraceGraphExecutor:
    graph: SubGraph
    reference_counts: Dict[TensorNode, int]
    shapes: Dict[TensorNode, Tuple[()]]

    def __init__(self, graph: SubGraph) -> None:
        self.graph = graph
        counts = defaultdict(lambda: 0)
        shapes = {}

        def record(node_ref):
            assert isinstance(node_ref, NodeRef)
            counts[node_ref] += 1
            node = node_ref()
            if isinstance(node, ShapeNode):
                shapes[node.source().ref] = ()

        for operator in graph.operators:
            for arg in operator().args:
                record(arg)
        for output in graph.outputs:
            record(output().ref)
        self.reference_counts = counts
        self.shapes = shapes

    def execute(self, *args: Tensor):
        assert len(args) == len(self.graph.inputs)
        values = {
            # node(): value for node, value in zip(self.graph.inputs, args) if self.reference_counts.get(node, 0) != 0
        }
        reference_counts = {}
        shapes = {}

        for node_ref, value in zip(self.graph.inputs, args):
            assert isinstance(node_ref, NodeRef)
            if self.reference_counts.get(node_ref, 0) != 0:
                values[node_ref] = value

        def consume_value(node: TensorNode):
            if isinstance(node, ConstantNode):
                return node.value
            if isinstance(node, ShapeNode):
                return shapes[node.source().ref][node.axis]
            value = values[node.ref]
            reference_counts[node.ref] -= 1
            if reference_counts[node.ref] == 0:
                del reference_counts[node.ref]
                del values[node.ref]
            return value
        
        def produce_value(node: TensorNode, value: Tensor):
            assert isinstance(node, NodeBase)
            values[node.ref] = value
            reference_counts[node.ref] = self.reference_counts[node.ref]
            if node.ref in self.shapes:
                shapes[node.ref] = value.shape

        for node, value in zip(self.graph.inputs, args):
            produce_value(node(), value)

        for operator in self.graph.operators:
            args = tuple(map(lambda arg: consume_value(arg()), operator().args))
            outputs = dispatch(operator().operator, *args)
            for node, output in zip(operator().outputs, outputs):
                produce_value(node(), output)

        outputs = tuple(map(lambda arg: consume_value(arg()), self.graph.outputs))
        return outputs
