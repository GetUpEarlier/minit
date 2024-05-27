from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple
import nvtx

from ..core.operator import Operator

from ..core.dispatch import dispatch

from ..core.tensor import Tensor
from ..graph import ConstantNode, OperatorNode, SubGraph, TensorNode, Use, ValueNode


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
    value_and_refs: Dict[ValueNode, ValueAndRefCount]
    sequence: List[Tuple[List[Callable[[], Tensor]], Operator, List[Callable[[Tensor], None]]]]

    def make_consumer(self, node_use: Use) -> Callable[[], Tensor]:
        if node_use.axis is not None:
            node = node_use.target
            axis = node_use.axis
            source = self.value_and_refs[node]
            return lambda: source.shape[axis]
        else:
            if isinstance(node_use.target, ConstantNode):
                constant = node_use.target.value
                return lambda: constant
            else:
                return self.value_and_refs[node_use.target].consume

    def make_producer(self, node: ValueNode) -> Callable[[Tensor], None]:
        value_and_ref = self.value_and_refs.get(node, None)
        if value_and_ref is not None:
            return value_and_ref.produce
        else:
            return lambda x: None

    def __init__(self, graph: SubGraph) -> None:
        self.graph = graph
        counts: DefaultDict[ValueNode, int] = defaultdict(lambda: 0)
        shapes: Dict[ValueNode, Tuple[ValueNode, ...]] = {}

        def record(node_use: Use):
            assert isinstance(node_use, Use)
            node = node_use()
            counts[node] += 1
            if node_use.axis is not None:
                shapes[node] = ()

        for operator in graph.operators:
            for arg in operator.args:
                record(arg)
        for output in graph.outputs:
            record(output)
        self.value_and_refs: Dict[ValueNode, ValueAndRefCount] = {
            node: ValueAndRefCount(ref_count, (node in shapes)) for node, ref_count in counts.items()
        }
        self.sequence = []
        for operator in graph.operators:
            self.sequence.append((
                [self.make_consumer(arg) for arg in operator.args],
                operator.operator,
                [self.make_producer(node) for node in operator.outputs],
            ))

    def execute(self, *args: Tensor):
        assert len(args) == len(self.graph.inputs)

        for node_use, value in zip(self.graph.inputs, args):
            assert node_use.axis is None
            self.make_producer(node_use.target)(value)

        for operator_inputs, operator, operator_outputs in self.sequence:
            operator_input_values = [operator_input() for operator_input in operator_inputs]
            operator_output_values = dispatch(operator, *operator_input_values)
            for operator_output, operator_output_value in zip(operator_outputs, operator_output_values):
                operator_output(operator_output_value)

        outputs = tuple([self.make_consumer(output)() for output in self.graph.outputs])
        return outputs

    def __call__(self, *args: Tensor):
        return self.execute(*args)
