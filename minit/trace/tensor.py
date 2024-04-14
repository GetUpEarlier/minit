from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union
from typing_extensions import Self
import weakref
from ..core.operator import Operator
from ..core.tensor import Tensor


from ..graph import GraphBuilder, NodeRef, ShapeNode, TensorNodeRef

class TraceTensor(Tensor):
    _value: Tensor
    _builder: GraphBuilder
    _node: TensorNodeRef

    def __init__(self, builder: GraphBuilder, node: TensorNodeRef, value: Tensor) -> None:
        super().__init__()
        self._builder = builder
        self._node = node
        self._value = value
        assert isinstance(node, NodeRef)

    @property
    def shape(self) -> Tuple[Tensor, ...]:
        return tuple(TraceTensor(self._builder, ShapeNode(self._node.graph, self._node, i).ref, dim) for i, dim in enumerate(self._value.shape))

    @property
    def dtype(self) -> str:
        return self._value.dtype

