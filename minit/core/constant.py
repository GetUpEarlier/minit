from numbers import Number
from typing import Tuple

from .dtype import dtype_info
from .tensor import Tensor


class ConstantTensor(Tensor):
    __slots__ = [
        "_value",
        "_shape",
        "_dtype",
    ]

    def __init__(self, value: Number, shape: Tuple[Tensor, ...], dtype: str) -> None:
        super().__init__()
        assert not isinstance(value, Tensor)
        assert shape is not None
        self._value = value
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype
    
    def value(self):
        return self._value

    def item(self) -> Number:
        assert len(self.shape) == 0
        return dtype_info(self._dtype).python_type(self._value)

    def __repr__(self) -> str:
        return f"Constant({self._value}, {self._shape}, {self._dtype})"

    def type(self):
        return ConstantTensor
