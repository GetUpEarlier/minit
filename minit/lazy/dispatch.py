from typing import Union

from ..core.meta import MetaTensor
from ..core.dispatch import dispatch, register_dispatch
from .tensor import Expression, LazyTensor
from ..core.scalar import ScalarTensor
from ..core.operator import Operator
from ..core.device_operator import DeviceOperator


def any_lazy(*tys):
    return any(ty in (LazyTensor, ScalarTensor) for ty in tys)


def any_lazy_but_some(*tys):
    return any(ty is LazyTensor for ty in tys) and any(ty not in (ScalarTensor, LazyTensor) for ty in tys)


@register_dispatch(predicate=any_lazy, priority=-2) 
def dispatch_lazy(op: Operator, *args: Union[ScalarTensor, LazyTensor]):
    meta_outputs = dispatch(DeviceOperator(op, "meta"), *[MetaTensor(arg.shape, arg.dtype) if isinstance(arg, LazyTensor) else arg for arg in args])
    expression = Expression(op, args, meta_outputs)
    outputs = tuple(LazyTensor(expression, i) for i in range(len(meta_outputs)))
    return outputs


@register_dispatch(predicate=any_lazy_but_some, priority=-2)
def dispatch_lazy_decay(op: Operator, *args: Union[ScalarTensor, LazyTensor]):
    meta_outputs = dispatch(op, *[MetaTensor(arg.shape, arg.dtype) for arg in args])
    expression = Expression(op, args, meta_outputs)
    outputs = tuple(LazyTensor(expression, i) for i in range(len(meta_outputs)))
    return outputs
