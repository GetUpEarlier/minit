from typing import Tuple, Union

from .tensor import Tensor
from .scalar import ScalarTensor


Shape = Tuple[Union[int, Tensor], ...]
ImmediateShape = Tuple[int, ...]
SymbolicShape = Tuple[Tensor, ...]


def to_immediate_shape(shape: Shape) -> ImmediateShape:
    return tuple(map(lambda dim: dim.item() if isinstance(dim, Tensor) else dim, shape))


def to_symbolic_shape(shape: Shape) -> SymbolicShape:
    return tuple(map(lambda dim: ScalarTensor(dim, "int32") if not isinstance(dim, Tensor) else dim, shape))
