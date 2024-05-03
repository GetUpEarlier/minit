from dataclasses import dataclass

from ..core.meta import MetaTensor
from ..core.tensor import Tensor
from ..core.dispatch import register_dispatch
from ..core.operator import Operator


@dataclass
class Fold(Operator):
    start: int
    stop: int

@dataclass
class Expand(Operator):
    axis: int

@dataclass
class AddAxis(Operator):
    axis: int

@dataclass
class RemoveAxis(Operator):
    axis: int

@dataclass
class Broadcast(Operator):
    axis: int

@dataclass
class Transpose(Operator):
    axis_a: int
    axis_b: int

@dataclass
class Reinterpret(Operator):
    target: str

@register_dispatch(priority=-1)
def dispatch_add_axis(op: AddAxis, x: Tensor):
    from ..functional.arith import constant
    z = MetaTensor(x.shape[:op.axis] + (constant(1, "int32"),) + x.shape[op.axis:], x.dtype)
    return (z,)

@register_dispatch(priority=-1)
def dispatch_broadcast(op: Broadcast, x: Tensor, size: Tensor):
    z = MetaTensor(x.shape[:op.axis] + (size,) + x.shape[op.axis+1:], x.dtype)
    return (z,)

@register_dispatch(priority=-1)
def dispatch_expand(op: Expand, x: Tensor, *sizes: Tensor):
    z = MetaTensor(sizes, x.dtype)
    return (z,)
