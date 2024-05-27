from dataclasses import dataclass
from numbers import Number
from typing import Literal

from ..core.tensor import Tensor
from ..core.device_operator import DeviceOperator
from ..core.dispatch import register_dispatch
from ..core.meta import MetaTensor
from ..core.scalar import ScalarTensor
from ..core.operator import Operator


@dataclass
class GenerateInterval(Operator):
    pass


@dataclass
class GenerateSequence(Operator):
    pass


@dataclass
class Fill(Operator):
    pass


@register_dispatch(priority=0)
def dispatch_generate_interval(op: DeviceOperator[GenerateInterval, Literal["meta"]], start: ScalarTensor, stop: ScalarTensor, step: ScalarTensor):
    size = stop.value() - start.value() / step.value()
    z = MetaTensor((ScalarTensor(size, (), "int32"),), start.dtype)
    return (z,)

@register_dispatch()
def register_fill(op: DeviceOperator[Fill, Literal["meta"]], value: Tensor, *sizes: Tensor):
    z = MetaTensor(sizes, value.dtype)
    return (z,)

@register_dispatch()
def register_fill(op: Fill, value: MetaTensor, *sizes: Tensor):
    z = MetaTensor(sizes, value.dtype)
    return (z,)

@register_dispatch()
def register_generate_sequence(op: DeviceOperator[GenerateSequence, Literal["meta"]], start: Tensor, size: Tensor, stop: Tensor):
    z = MetaTensor((size,), start.dtype)
    return (z,)

@register_dispatch()
def register_generate_sequence(op: GenerateSequence, start: MetaTensor, size: Tensor, stop: Tensor):
    z = MetaTensor((size,), start.dtype)
    return (z,)
