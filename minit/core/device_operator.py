from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from .dispatch import dispatch, register_dispatch
from .operator import Operator
from .tensor import Tensor

_Operator = TypeVar("_Operator", bound=Operator)
_Device = TypeVar("_Device", bound=str)


@dataclass
class DeviceOperator(Operator, Generic[_Operator, _Device]):
    operator: _Operator
    device: _Device

    def type(self):
        return DeviceOperator[self.operator.type(), Literal[self.device]] # type: ignore


@register_dispatch(priority=-2)
def register_device_dispatch(op: DeviceOperator[Operator, Literal["auto"]], *args: "Tensor"):
    return dispatch(op.operator, *args)