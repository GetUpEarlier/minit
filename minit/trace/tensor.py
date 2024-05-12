from typing import Tuple
from ..core.tensor import Tensor


from ..graph import GraphBuilder, Use, ShapeUse, ValueUse

class TraceTensor(Tensor):
    _value: Tensor
    _builder: GraphBuilder
    _node: Use

    def __init__(self, builder: GraphBuilder, node: Use, value: Tensor) -> None:
        super().__init__()
        self._builder = builder
        self._node = node
        self._value = value
        assert isinstance(node, (ShapeUse, ValueUse))

    @property
    def shape(self) -> Tuple[Tensor, ...]:
        return tuple(TraceTensor(self._builder, self._node.shape(i), dim) for i, dim in enumerate(self._value.shape))

    @property
    def dtype(self) -> str:
        return self._value.dtype

    @property
    def device(self):
        return "trace"
