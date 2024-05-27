from dataclasses import dataclass

from ..core.meta import MetaTensor
from ..core.dispatch import register_dispatch
from ..core.operator import Operator


@dataclass
class Slice(Operator):
    axis: int

class SliceSet(Operator):
    ...

@dataclass
class Index(Operator):
    axis: int

class IndexSet(Operator):
    ...

@dataclass
class Split(Operator):
    axis: int

@dataclass
class Tie(Operator):
    axis: int

@register_dispatch()
def register_tie(op: Tie, *args: MetaTensor):
    for arg in args[1:]:
        assert arg.dtype == args[0].dtype
        assert len(arg.shape) == len(args[0].shape)
    b = args[0].shape[op.axis]
    for arg in args[1:]:
        b = b + arg.shape[op.axis]
    z = MetaTensor(args[0]._shape[:op.axis] + (b,) + args[0].shape[op.axis+1:], args[0].dtype)
    return (z,)
