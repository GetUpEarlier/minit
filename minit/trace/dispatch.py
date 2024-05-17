from typing import List, Literal, Union

from ..core.meta import MetaTensor
from ..lazy.tensor import Expression, LazyTensor
from ..core.tensor import Tensor
from .tensor import TraceTensor
from ..graph import GraphBuilder, Use
from ..core.operator import Operator
from ..core.device_operator import DeviceOperator
from ..core.dispatch import dispatch, register_dispatch
from ..core.object import match_pattern


def any_trace_tensor(*args):
    for arg in args:
        if match_pattern(TraceTensor, arg):
            return True
    return False


def _dispatch_trace(op: Operator, *args: Union[TraceTensor, Tensor]):
    arg_values = []
    arg_nodes: List[Use] = []
    builder = None
    for arg in args:
        if isinstance(arg, TraceTensor):
            if builder is None:
                builder = arg._builder
            else:
                assert builder == arg._builder
    assert builder is not None
    for arg in args:
        if isinstance(arg, LazyTensor):
            arg = trace_evaluate(builder, arg._expression)[arg._index]
            assert not isinstance(arg, LazyTensor)
        if isinstance(arg, TraceTensor):
            arg_values.append(arg._value)
            arg_nodes.append(arg._node)
        else:
            arg_values.append(arg)
            arg_nodes.append(builder.create_constant(arg))
    output_values = dispatch(op, *arg_values)
    output_metas = tuple(MetaTensor(output_value.shape, output_value.dtype) for output_value in output_values)
    output_uses = builder.create_operator(op, arg_nodes, output_metas)
    return tuple(TraceTensor(builder, output_use, output_value) for output_use, output_value in zip(output_uses, output_values))


def trace_evaluate(builder: GraphBuilder, expression: Expression):
    args = tuple([trace_evaluate(builder, arg._expression)[arg._index] if isinstance(arg, LazyTensor) else TraceTensor(builder, builder.create_constant(arg), arg) for arg in expression.args])
    return dispatch(expression.op, *args)


@register_dispatch(predicate=any_trace_tensor, priority=1)
def dispatch_any(op: Operator, *args: Union[TraceTensor, Tensor]):
    return _dispatch_trace(op, *args)


@register_dispatch(priority=1)
def register_device_operator(op: DeviceOperator[Operator, Literal["trace"]], *args: Union[TraceTensor, Tensor]):
    return _dispatch_trace(op.operator, *args)
