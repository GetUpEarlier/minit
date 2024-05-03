from typing import Optional, Sequence, Tuple

from .shape import to_symbolic_shape
from .tensor import Tensor


class MetaTensor(Tensor):
    def __init__(self, shape: Tuple[Tensor, ...], dtype: str) -> None:
        super().__init__()
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self):
        return to_symbolic_shape(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @staticmethod
    def make(shape: Sequence[Optional[Tensor]], dtype: str):
        shape = tuple([dim if dim is not None else MetaTensor((), "int32") for dim in shape])
        return MetaTensor(shape, dtype)

    @property
    def device(self):
        return "meta"
