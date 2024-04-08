from typing import Tuple, Union

from .tensor import Tensor
from .scalar import ScalarTensor


Shape = Tuple[Union[int, ScalarTensor], ...]
ImmediateShape = Tuple[int, ...]


def to_immediate_shape(shape: Shape) -> ImmediateShape:
    return tuple(map(lambda dim: dim.item() if isinstance(dim, Tensor) else dim, shape))
