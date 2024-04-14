from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union
import weakref
from typing_extensions import Self

from ..core.operator import Operator

from ..core.tensor import Tensor


_Node = TypeVar("_Node", covariant=True)


@dataclass(frozen=True)
class GraphRef:
    if TYPE_CHECKING:
        graph: weakref.ref["Graph"]
    else:
        graph: weakref.ref

    def __call__(self) -> "Graph":
        graph = self.graph()
        assert graph is not None
        return graph

    @property
    def valid(self):
        return self.graph() is not None

    def __post_init__(self):
        assert isinstance(self.graph, weakref.ref)


@dataclass(frozen=True)
class NodeRef(Generic[_Node]):
    graph: GraphRef
    id: int

    def __call__(self) -> _Node:
        return self.graph().nodes[self.id]
    
    @property
    def valid(self):
        return self.graph.valid
    
    def __repr__(self) -> str:
        return f"NodeRef({self.id})"


class NodeBase:
    graph: "GraphRef"
    id: int

    def __init__(self, graph: "GraphRef") -> None:
        self.graph = graph
        nodes = graph().nodes
        id = len(nodes)
        nodes.append(self)
        self.id = id

    @property
    def ref(self) -> NodeRef[Self]:
        return NodeRef(self.graph, self.id)
    

class Use():
    def __init__(self, value: NodeRef["ValueNode"]) -> None:
        assert isinstance(value, NodeRef)
        self.value = value

    def __call__(self):
        return self.value()

    def __del__(self):
        if self.value.valid:
            self.value().uses.remove(weakref.ref(self))

    def __repr__(self) -> str:
        return f"Use{{{self.value}}}"


class ValueNode(NodeBase):
    if TYPE_CHECKING:
        uses: List[weakref.ref[Use]]
    else:
        uses: List[weakref.ref]

    def __init__(self, graph: GraphRef) -> None:
        super().__init__(graph)
        self.uses = []

    def use(self):
        use = Use(self.ref)
        self.uses.append(weakref.ref(use))
        return use

    def replace(self, new: NodeRef["ValueNode"]):
        assert isinstance(new, NodeRef)
        uses = self.uses
        for use in uses:
            use().value = new
        new().uses.extend(uses)
        assert self.uses is uses
        self.uses.clear()


class OperatorNode(NodeBase):
    operator: Operator
    args: List["TensorNodeRef"]
    outputs: Tuple[Use, ...]

    def __init__(self, graph: "GraphRef", operator: Operator, args: Tuple["TensorNodeRef", ...], nr_outputs: int) -> None:
        super().__init__(graph)
        self.operator = operator
        self.args = list(args)
        self.outputs = tuple(map(lambda i: InternalNode(graph, self.ref, i).use(), range(nr_outputs)))

    def __repr__(self) -> str:
        return repr({
            "operator": self.operator,
            "args": self.args,
            "outputs": self.outputs,
        })


class InternalNode(ValueNode):
    producer: "OperatorNodeRef"
    index: int

    def __init__(self, graph: "GraphRef", producer: "OperatorNodeRef", index: int):
        super().__init__(graph)
        self.graph = graph
        self.producer = producer
        self.index = index


class ConstantNode(ValueNode):
    value: Tensor

    def __init__(self, graph: "GraphRef", value: Tensor):
        super().__init__(graph)
        self.value = value


class PlaceholderNode(ValueNode):
    def __init__(self, graph: "GraphRef"):
        super().__init__(graph)
        graph().placeholders.append(self)


class ShapeNode(ValueNode):
    source: Use
    axis: int

    def __init__(self, graph: "GraphRef", source: "TensorNodeRef", axis: int):
        super().__init__(graph)
        self.source = source().use()
        self.axis = axis


TensorNode = Union[InternalNode, ConstantNode, PlaceholderNode, ShapeNode]
TensorNodeRef = NodeRef[TensorNode]
OperatorNodeRef = NodeRef[OperatorNode]
Node = Union[TensorNode, OperatorNode]


class Graph:
    placeholders: List[NodeRef[PlaceholderNode]]
    nodes: List[Node]

    def __init__(self) -> None:
        self.placeholders = []
        self.nodes = []

    @property
    def ref(self):
        return GraphRef(weakref.ref(self))


_T = TypeVar("_T")

class LinkedList(Generic[_T]):
    class Node(Generic[_T]):
        value: Optional[_T]
        if TYPE_CHECKING:
            prev: Optional[weakref.ref["LinkedList.Node[_T]"]] = None
        else:
            prev: Optional[weakref.ref] = None
        next: Optional["LinkedList.Node[_T]"] = None

        def __init__(self, value: Optional[_T]) -> None:
            self.value = value

    def __init__(self) -> None:
        self.head = LinkedList.Node(None)
        self.tail = LinkedList.Node(None)
        self.head.next = self.tail
        self.tail.prev = weakref.ref(self.head)

    def append(self, value: _T):
        assert value is not None
        node = LinkedList.Node(value)
        assert self.tail.prev is not None
        tail_prev = self.tail.prev()
        assert tail_prev is not None
        tail_prev.next = node
        self.tail.prev = weakref.ref(node)
        node.next = self.tail
        node.prev = weakref.ref(tail_prev)

    def __iter__(self):
        node = self.head.next
        while node != self.tail:
            yield node.value
            node = node.next

    def check(self):
        list(self)

    head: Node[_T]
    tail: Node[_T]


class SubGraph:
    graph: Graph
    inputs: List[TensorNodeRef]
    operators: LinkedList[OperatorNodeRef]
    outputs: List[Use]

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.inputs = []
        self.operators = LinkedList()
        self.outputs = []

    def replace(self, old: "SubGraph", new: "SubGraph"):
        assert old.inputs == new.inputs
        node = old.operators.head
        while node.next != old.operators.tail:
            node = node.next
        assert new.operators.head.next != new.operators.tail
        old.operators.tail.prev = new.operators.tail.prev
        old.operators.tail.prev().next = old.operators.tail
        old.operators.head.next = new.operators.head.next
        old.operators.head.next.prev = weakref.ref(old.operators.head)
        for old_output, new_output in zip(old.outputs, new.outputs):
            old_output().replace(new_output().ref)

    def __repr__(self):
        return repr({
            "inputs": self.inputs,
            "operators": list(self.operators),
            "outputs": self.outputs,
        })

    def values(self) -> Iterable[NodeRef["ValueNode"]]:
        for input_ref in self.inputs:
            yield input_ref
        for output_ref in self.outputs:
            yield output_ref

    def replace_value(self, old: NodeRef["ValueNode"], new: NodeRef["ValueNode"]):
        for i in range(len(self.inputs)):
            if self.inputs[i] == old:
                self.inputs[i] = new
        for i in range(len(self.outputs)):
            if self.outputs[i] == old:
                self.outputs[i] = new


class GraphBuilder:
    graph: SubGraph

    def __init__(self, graph: Graph) -> None:
        self.graph = SubGraph(graph)

    def create_input(self):
        input = PlaceholderNode(self.graph.graph.ref).ref
        self.graph.inputs.append(input)
        return input

    def create_operator(self, operator: Operator, args: Tuple[TensorNodeRef, ...], nr_outputs: int):
        operator_node = OperatorNode(self.graph.graph.ref, operator, args, nr_outputs)
        self.graph.operators.append(operator_node.ref)
        return tuple(map(lambda output: output().ref, operator_node.outputs))

    def create_constant(self, value: Tensor):
        constant = ConstantNode(self.graph.graph.ref, value)
        return constant.ref

    def build(self, *outputs: TensorNodeRef) -> SubGraph:
        self.graph.outputs = list(map(lambda output: output().use(), outputs))
        return self.graph
