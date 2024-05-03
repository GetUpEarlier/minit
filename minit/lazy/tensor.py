from dataclasses import dataclass
from typing import Tuple

from ..core.device_operator import DeviceOperator
from ..core.dispatch import dispatch
from ..core.operator import Operator
from ..core.tensor import Tensor


@dataclass
class Expression:
    op: Operator
    args: Tuple["Tensor", ...]
    outputs: Tuple["Tensor", ...]

    def evaluate(self, device: str) -> Tuple["Tensor", ...]:
        args = tuple([arg.instantiate(device) if isinstance(arg, LazyTensor) else arg for arg in self.args])
        return dispatch(DeviceOperator(self.op, device), *args)


class LazyTensor(Tensor):
    def __init__(self, expression: Expression, index: int) -> None:
        super().__init__()
        self._expression = expression
        self._index = index

    @property
    def dtype(self):
        return self._expression.outputs[self._index].dtype

    @property
    def shape(self):
        return self._expression.outputs[self._index].shape

    @property
    def device(self):
        return "lazy"

    def instantiate(self, device: str):
        return self._expression.evaluate(device)[self._index]
