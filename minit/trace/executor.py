from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import nvtx

from ..core.operator import Operator

from ..core.dispatch import dispatch

from ..core.tensor import Tensor
from ..graph import ConstantNode, NodeRef, OperatorNode, ShapeNode, SubGraph, TensorNode


class ValueAndRefCount:
    __slots__ = [
        "value", "ref_count", "init_ref_count", "require_shape", "shape",
    ]

    value: Optional[Tensor]
    ref_count: int
    init_ref_count: int
    require_shape: bool
    shape: Optional[Tuple[Tensor, ...]]

    def __init__(self, init_ref_count: int, require_shape: bool) -> None:
        self.value = None
        self.ref_count = 0
        self.init_ref_count = init_ref_count
        self.require_shape = require_shape
        self.shape = None

    def consume(self) -> Tensor:
        value = self.value
        assert value is not None
        self.ref_count -= 1
        assert self.ref_count >= 0
        if self.ref_count == 0:
            self.value = None
        return value

    def produce(self, value: Tensor):
        self.ref_count = self.init_ref_count
        if self.init_ref_count != 0:
            self.value = value
        if self.require_shape:
            self.shape = value.shape


class TraceGraphExecutor:
    graph: SubGraph
    value_and_refs: Dict[NodeRef, ValueAndRefCount]
    sequence: List[Tuple[List[Callable[[], Tensor]], Operator, List[Callable[[Tensor], None]]]]

    def make_consumer(self, node_ref: NodeRef[TensorNode]):
        if node_ref.type is ConstantNode:
            constant = node_ref().value
            return lambda: constant
        elif node_ref.type is ShapeNode:
            node = node_ref()
            source = self.value_and_refs[node.source.value]
            axis = node.axis
            return lambda: source.shape[axis]
        else:
            return self.value_and_refs[node_ref].consume

    def make_producer(self, node_ref: NodeRef[TensorNode]):
        value_and_ref = self.value_and_refs.get(node_ref, None)
        if value_and_ref is not None:
            return value_and_ref.produce
        else:
            return lambda x: None

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
            record(output.value)
        self.value_and_refs = {
            node_ref: ValueAndRefCount(ref_count, (node_ref in shapes)) for node_ref, ref_count in counts.items()
        }
        self.sequence = []
        for operator_ref in graph.operators:
            operator: OperatorNode = operator_ref()
            self.sequence.append((
                [self.make_consumer(arg) for arg in operator.args],
                operator.operator,
                [self.make_producer(node_use.value) for node_use in operator.outputs],
            ))

    def execute(self, *args: Tensor):
        assert len(args) == len(self.graph.inputs)

        for node_ref, value in zip(self.graph.inputs, args):
            self.make_producer(node_ref)(value)

        for operator_inputs, operator, operator_outputs in self.sequence:
            operator_input_values = [operator_input() for operator_input in operator_inputs]
            operator_output_values = dispatch(operator, *operator_input_values)
            for operator_output, operator_output_value in zip(operator_outputs, operator_output_values):
                operator_output(operator_output_value)

        outputs = tuple([self.make_consumer(output.value)() for output in self.graph.outputs])
        return outputs
