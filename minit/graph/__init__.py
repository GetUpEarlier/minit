import builtins
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar, Union
import weakref
from typing_extensions import Self
import nvtx
import graphviz

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


class NodeRef(Generic[_Node]):
    __slots__ = [
        "graph",
        "id",
        "type",
    ]

    graph: GraphRef
    id: int
    type: Type[_Node]

    def __init__(self, graph: GraphRef, id: int, type: Type[_Node]):
        self.graph = graph
        self.id = id
        self.type = type

    def __call__(self) -> _Node:
        return self.graph().nodes[self.id]

    @property
    def valid(self):
        return self.graph.valid

    def __repr__(self) -> str:
        return f"NodeRef({self.id}, {self.type})"


class NodeBase:
    __slots__ = [
        "graph",
        "id",
        "ref",
    ]

    graph: "GraphRef"
    id: int
    ref: NodeRef[Self]

    def __init__(self, graph: "GraphRef") -> None:
        self.graph = graph
        nodes = graph().nodes
        id = len(nodes)
        nodes.append(self)
        self.id = id
        self.ref = NodeRef(self.graph, self.id, type(self))


class Use():
    __slots__ = [
        "value",
        "__weakref__",
    ]

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
    __slots__ = [
        "graph",
        "id",
        "ref",
        "uses",
    ]

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
    __slots__ = [
        "graph",
        "id",
        "ref",
        "operator",
        "args",
        "outputs",
    ]

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
    __slots__ = [
        "graph",
        "id",
        "ref",
        "uses",
        "producer",
        "index",
    ]

    producer: "OperatorNodeRef"
    index: int

    def __init__(self, graph: "GraphRef", producer: "OperatorNodeRef", index: int):
        super().__init__(graph)
        self.producer = producer
        self.index = index


class ConstantNode(ValueNode):
    __slots__ = [
        "graph",
        "id",
        "ref",
        "uses",
        "value",
    ]

    value: Tensor

    def __init__(self, graph: "GraphRef", value: Tensor):
        super().__init__(graph)
        self.value = value


class PlaceholderNode(ValueNode):
    def __init__(self, graph: "GraphRef"):
        super().__init__(graph)
        graph().placeholders.append(self)


class ShapeNode(ValueNode):
    __slots__ = [
        "graph",
        "id",
        "ref",
        "uses",
        "source",
        "axis",
    ]

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
    __slots__ = [
        "placeholders",
        "nodes",
        "__weakref__",
    ]

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
    __slots__ = [
        "head",
        "tail",
    ]

    class Node(Generic[_T]):
        __slots__ = [
            "value",
            "prev",
            "next",
            "__weakref__",
        ]

        value: Optional[_T]
        if TYPE_CHECKING:
            prev: Optional[weakref.ref["LinkedList.Node[_T]"]]
        else:
            prev: Optional[weakref.ref]
        next: Optional["LinkedList.Node[_T]"]

        def __init__(self, value: Optional[_T]) -> None:
            self.value = value
            self.prev = None
            self.next = None

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

    def tolist(self) -> List[_T]:
        result = []
        node = self.head.next
        while node is not self.tail:
            result.append(node.value)
            node = node.next
        return result

    def __iter__(self):
        node = self.head.next
        while node is not self.tail:
            yield node.value
            node = node.next

    def check(self):
        list(self)

    head: Node[_T]
    tail: Node[_T]


class SubGraph:
    __slots__ = [
        "graph",
        "inputs",
        "operators",
        "outputs",
    ]

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
    

def dump_graphviz(subgraph: SubGraph):
    dot = graphviz.Digraph('round-table', comment='The Round Table')
    nodes = {}

    def create_node(node_ref: NodeRef):
        if node_ref not in nodes:
            prefix = node_ref.type.__name__
            dot.node(f"{prefix}_{node_ref.id}")
            nodes[node_ref] = f"{prefix}_{node_ref.id}"
        return nodes[node_ref]

    def create_edge(source: NodeRef, target: NodeRef):
        dot.edge(create_node(source), create_node(target))

    for input in subgraph.inputs:
        create_node(input)

    for operator_ref in subgraph.operators:
        operator = operator_ref()
        for operator_input in operator.args:
            if operator_input.type is ShapeNode:
                create_edge(operator_input().source.value, operator_input)
            create_edge(operator_input, operator_ref)
        for operator_output in operator.outputs:
            create_edge(operator_ref, operator_output.value)

    for output in subgraph.outputs:
        create_node(output.value)

    return dot
