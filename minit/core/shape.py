from typing import Tuple, Union

from .tensor import Tensor
from .scalar import ScalarTensor


Shape = Tuple[Union[int, Tensor], ...]
ImmediateShape = Tuple[int, ...]
SymbolicShape = Tuple[Tensor, ...]


def to_immediate_shape(shape: Shape) -> ImmediateShape:
    return tuple([dim.item() if isinstance(dim, Tensor) else dim for dim in shape])


def to_symbolic_shape(shape: Shape) -> SymbolicShape:
    return tuple([dim if isinstance(dim, Tensor) else ScalarTensor(dim, (), "int32") for dim in shape])
