from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union
import weakref

from ..core.scalar import ScalarTensor
from ..core.meta import MetaTensor
from ..core.operator import Operator
from ..core.tensor import Tensor


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


class NodeBase:
    __slots__ = [
        "graph",
        "valid",
    ]

    graph: "GraphRef"
    valid: bool

    def __init__(self, graph: "GraphRef") -> None:
        self.graph = graph
        self.valid = True


# class Use:
#     value: "ValueNode"

#     def __init__(self, value: "ValueNode") -> None:
#         assert isinstance(value, ValueNode)
#         self.value = value

#     def __call__(self):
#         return self.value

#     def __del__(self):
#         if self.value.valid:
#             self.value.uses.remove(weakref.ref(self))


class ValueUse:
    target: "ValueNode"
    user: Optional["OperatorNode"]
    axis = None

    def __init__(self, target: "ValueNode", user: Optional["OperatorNode"]) -> None:
        assert isinstance(target, ValueNode)
        self.target = target
        self.user = user

    def clone(self, user: "OperatorNode"):
        return self.target.use_value(user)
    
    def shape(self, axis: int):
        return self.target.use_shape(None, axis)

    def __call__(self):
        return self.target

    def __del__(self):
        if self.target.valid:
            self.target.uses.remove(weakref.ref(self))

    def __repr__(self):
        return f"{self.target}.value"


class ShapeUse:
    target: "ValueNode"
    user: Optional["OperatorNode"]
    axis: int

    def __init__(self, target: "ValueNode", user: Optional["OperatorNode"], axis: int) -> None:
        assert isinstance(target, ValueNode)
        self.target = target
        self.user = user
        self.axis = axis

    def clone(self, user: "OperatorNode"):
        return self.target.use_shape(user, self.axis)

    def __call__(self):
        return self.target

    def __del__(self):
        if self.target.valid:
            self.target.uses.remove(weakref.ref(self))

    def __repr__(self):
        return f"{self.target}.shape[{self.axis}]"


Use = Union[ValueUse, ShapeUse]


class ValueNode(NodeBase):
    if TYPE_CHECKING:
        uses: List[weakref.ref[Use]]
    else:
        uses: List[weakref.ref]

    def __init__(self, graph: GraphRef) -> None:
        super().__init__(graph)
        self.uses = []

    def use_shape(self, user: Optional["OperatorNode"], axis: int):
        use = ShapeUse(self, user, axis)
        self.uses.append(weakref.ref(use))
        return use

    def use_value(self, user: Optional["OperatorNode"]):
        if isinstance(self, InternalNode):
            assert user != self.producer
        use = ValueUse(self, user)
        self.uses.append(weakref.ref(use))
        return use
    
    def replace(self, target: "ValueNode"):
        if self is target:
            return
        for use in self.uses:
            use().target = target
        target.uses += self.uses
        self.uses.clear()
        self.valid = False


class OperatorNode(NodeBase):
    __slots__ = [
        "graph",
        "valid",
        "operator",
        "args",
        "outputs",
    ]

    operator: Operator
    args: List[Use]
    outputs: Tuple["InternalNode", ...]

    def __init__(self, graph: "GraphRef", operator: Operator, args: Sequence[Use], output_metas: Tuple[MetaTensor, ...]) -> None:
        super().__init__(graph)
        self.operator = operator
        self.args = [arg.clone(self) for arg in args]
        self.outputs = tuple(InternalNode(graph, self, i, meta) for i, meta in enumerate(output_metas))

    def destroy(self):
        del self.operator
        del self.args
        del self.outputs
        self.valid = False

    def __repr__(self) -> str:
        return repr({
            "operator": self.operator,
            "args": self.args,
            "outputs": self.outputs,
        })


class InternalNode(ValueNode):
    __slots__ = [
        "graph",
        "valid",
        "uses",
        "producer",
        "index",
        "meta",
    ]

    producer: "OperatorNode"
    index: int
    meta: MetaTensor

    def __init__(self, graph: "GraphRef", producer: "OperatorNode", index: int, meta: MetaTensor):
        super().__init__(graph)
        self.producer = producer
        self.index = index
        self.meta = meta

    @property
    def shape(self):
        return self.meta.shape
    
    @property
    def dtype(self):
        return self.meta.dtype
    
    def __repr__(self):
        return "InternalNode()"


class ConstantNode(ValueNode):
    __slots__ = [
        "graph",
        "valid",
        "uses",
        "value",
    ]

    value: Tensor

    def __init__(self, graph: "GraphRef", value: Tensor):
        super().__init__(graph)
        assert isinstance(value, ScalarTensor), f"{value} is not ScalarTensor"
        self.value = value

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype


class PlaceholderNode(ValueNode):
    __slots__ = [
        "graph",
        "valid",
        "uses",
        "value",
        "meta",
    ]

    def __init__(self, graph: "GraphRef", meta: MetaTensor):
        super().__init__(graph)
        self.meta = meta

    @property
    def shape(self):
        return self.meta.shape

    @property
    def dtype(self):
        return self.meta.dtype
    
    def __repr__(self):
        return "PlaceholderNode()"


# class ShapeNode(ValueNode):
#     __slots__ = [
#         "graph",
#         "valid",
#         "uses",
#         "source",
#         "axis",
#     ]

#     source: Use
#     axis: int

#     def __init__(self, graph: "GraphRef", source: "TensorNode", axis: int):
#         super().__init__(graph)
#         self.source = source().use()
#         self.axis = axis

#     @property
#     def shape(self):
#         return ()

#     @property
#     def dtype(self):
#         return "int32"


TensorNode = Union[InternalNode, ConstantNode, PlaceholderNode]
Node = Union[TensorNode, OperatorNode]


_T = TypeVar("_T")

class LinkedListVertex(Generic[_T]):
    prev: "LinkedListEdge[_T]"
    next: "LinkedListEdge[_T]"

class LinkedListEdge(Generic[_T]):
    prev: "LinkedListVertex[_T]"
    next: "LinkedListVertex[_T]"
    value: Optional[_T]

    def __init__(self, prev: "LinkedListVertex[_T]", next: "LinkedListVertex[_T]", value: Optional[_T] = None) -> None:
        super().__init__()
        self.prev = prev
        self.next = next
        self.value = value
        prev.next = self
        next.prev = self

class LinkedListIterator(Generic[_T]):
    current: LinkedListVertex[_T]
    tail: LinkedListVertex[_T]

    def __init__(self, head: LinkedListVertex[_T], tail: LinkedListVertex[_T]):
        self.current = head
        self.tail = tail

    def clone(self):
        return LinkedListIterator(self.current, self.tail)

    def __next__(self) -> _T:
        while self.current is not self.tail:
            value = self.current.next.value
            self.current = self.current.next.next
            if value is not None:
                return value
        raise StopIteration


class LinkedList(Generic[_T], Iterable[_T]):
    head: LinkedListVertex[_T]
    tail: LinkedListVertex[_T]

    def __init__(self):
        self.head = LinkedListVertex()
        self.tail = LinkedListVertex()
        LinkedListEdge(self.head, self.tail, None)
        LinkedListEdge(self.tail, self.head, None)

    def view(self):
        return LinkedListView(self.head, self.tail)

    def tolist(self) -> List[_T]:
        result = []
        node = self.head
        while node is not self.tail:
            if node.next.value is not None:
                result.append(node.next.value)
            node = node.next.next
        return result

    def __iter__(self) -> LinkedListIterator[_T]:
        return LinkedListIterator(self.head, self.tail)


class LinkedListView(Generic[_T], Iterable[_T]):
    head: LinkedListVertex[_T]
    tail: LinkedListVertex[_T]

    def __init__(self, head: LinkedListVertex[_T], tail: LinkedListVertex[_T]) -> None:
        super().__init__()
        assert head is not tail
        self.head = head
        self.tail = tail

    def tolist(self) -> List[_T]:
        result = []
        node = self.head
        while node is not self.tail:
            if node.next.value is not None:
                result.append(node.next.value)
            node = node.next.next
        return result
    
    def clear(self):
        node = self.head.next.next
        while node is not self.tail:
            next = node.next.next
            node.prev = None
            node.next = None
            node = next
        LinkedListEdge(self.head, self.tail, None)

    def fill(self, list: List[_T]):
        node = self.head.next.next
        while node is not self.tail:
            next = node.next.next
            node.prev = None
            node.next = None
            node = next
        if len(list) == 0:
            LinkedListEdge(self.head, self.tail, None)
        else:
            node = self.head
            for item in list:
                new_node = LinkedListVertex()
                LinkedListEdge(node, new_node, item)
                node = new_node

    def append(self, value: _T):
        node = LinkedListVertex()
        self.tail.prev.next = node
        node.prev = self.tail.prev
        LinkedListEdge(node, self.tail, value)

    def __iter__(self) -> LinkedListIterator[_T]:
        return LinkedListIterator(self.head, self.tail)


class Graph:
    __slots__ = [
        "inputs",
        "operators",
        "outputs",
        "__weakref__",
    ]

    inputs: List[PlaceholderNode]
    operators: "LinkedList[OperatorNode]"
    outputs: List[Use]

    def __init__(self) -> None:
        self.inputs = []
        self.operators = LinkedList()
        self.outputs = []

    def create_input(self, meta: MetaTensor):
        self.inputs.append(PlaceholderNode(self.ref, meta))
        return self.inputs[-1]

    @property
    def ref(self):
        return GraphRef(weakref.ref(self))


class SubGraph:
    __slots__ = [
        "graph",
        "inputs",
        "operators",
        "outputs",
    ]

    graph: Graph
    inputs: List[Use]
    operators: LinkedListView[OperatorNode]
    outputs: List[Use]

    def __init__(self, graph: Graph, inputs: Sequence[Use], operators: LinkedListView[OperatorNode], outputs: Sequence[Use]) -> None:
        self.graph = graph
        self.inputs = list(inputs)
        self.operators = operators
        self.outputs = list(outputs)

    def __repr__(self):
        return repr({
            "inputs": self.inputs,
            "operators": list(self.operators),
            "outputs": self.outputs,
        })


class GraphBuilder:
    graph: Graph
    inputs: List[Use]
    operators: LinkedListView[OperatorNode]

    def __init__(self, graph: Graph, inputs: List[Use], operators: LinkedListView[OperatorNode]) -> None:
        assert isinstance(graph, Graph)
        assert isinstance(operators, LinkedListView)
        self.graph = graph
        self.inputs = list(inputs)
        self.operators = operators

    def create_operator(self, operator: Operator, args: Sequence[Use], output_metas: Tuple[MetaTensor, ...]) -> Tuple[ValueNode, ...]:
        operator_node = OperatorNode(self.graph.ref, operator, args, output_metas)
        self.operators.append(operator_node)
        return tuple(output.use_value(None) for output in operator_node.outputs)

    def create_constant(self, value: Tensor):
        assert not isinstance(value, MetaTensor)
        constant = ConstantNode(self.graph.ref, value)
        return constant.use_value(None)

    def build(self, *outputs: Use) -> Tuple[Use, ...]:
        return outputs
