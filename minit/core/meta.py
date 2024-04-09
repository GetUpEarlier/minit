from typing import Tuple

from .shape import to_symbolic_shape
from .tensor import Tensor


class MetaTensor(Tensor):
    def __init__(self, shape: Tuple[int, ...], dtype: str) -> None:
        super().__init__()
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self):
        return to_symbolic_shape(self._shape)

    @property
    def dtype(self):
        return self._dtype
