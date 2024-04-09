from typing import Generic, TypeVar
from .functional import dequantize
from ..core.tensor import Tensor


_Data = TypeVar("_Data")


class QuantizedTensor(Tensor, Generic[_Data]):
    def __init__(self, data: Tensor, group: Tensor, zero: Tensor, scale: Tensor) -> None:
        super().__init__()
        self._data = data
        self._group = group
        self._zero = zero
        self._scale = scale

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return f"q{self._data.dtype}"

    def dequantize(self) -> Tensor:
        return dequantize(self._data, self._group, self._zero, self._scale, 0)