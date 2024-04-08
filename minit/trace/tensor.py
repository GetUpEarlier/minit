from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from ..core.operator import Operator
from ..core.tensor import Tensor


@dataclass
class TraceOperatorNode:
    operator: Operator
    args: Tuple["TraceNode", ...]
    outputs: Optional[Tuple["InternalNode", ...]] = None


class InternalNode:
    graph: "TraceGraph"
    operator_node: TraceOperatorNode
    index: int

    def __init__(self, graph: "TraceGraph", operator_node: TraceOperatorNode, index: int):
        self.graph = graph
        self.operator_node = operator_node
        self.index = index


class ConstantNode:
    graph: "TraceGraph"
    value: Tensor

    def __init__(self, graph: "TraceGraph", value: Tensor):
        self.graph = graph
        self.value = value


class InputNode:
    graph: "TraceGraph"
    index: int

    def __init__(self, graph: "TraceGraph", index: int):
        self.graph = graph
        self.index = index


@dataclass
class ShapeNode:
    graph: "TraceGraph"
    source: "TraceNode"
    axis: int


TraceNode = Union[InternalNode, ConstantNode, InputNode]


@dataclass
class TraceGraph:
    inputs: Tuple[InputNode, ...] = ()
    operators: List[TraceOperatorNode] = field(default_factory=list)
    outputs: Tuple[TraceNode, ...] = ()


class TraceTensor(Tensor):
    _value: Tensor
    _node: TraceNode

    def __init__(self, node: TraceNode, value: Tensor) -> None:
        super().__init__()
        self._node = node
        self._value = value

    @property
    def shape(self) -> Tuple[Tensor, ...]:
        return tuple(TraceTensor(ShapeNode(self._node.graph, self._node, i), dim) for i, dim in enumerate(self._value.shape))

    @property
    def dtype(self) -> str:
        return self._value.dtype

