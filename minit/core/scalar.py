from numbers import Number

from .dtype import dtype_info
from .tensor import Tensor


class ScalarTensor(Tensor):
    def __init__(self, value: Number, dtype: str) -> None:
        super().__init__()
        assert not isinstance(value, Tensor)
        self._value = value
        self._dtype = dtype

    @property
    def shape(self):
        return ()

    @property
    def dtype(self):
        return self._dtype

    def item(self) -> Number:
        return dtype_info(self._dtype).python_type(self._value)
