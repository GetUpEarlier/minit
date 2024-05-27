from numbers import Number
from typing import Callable, Sequence, Tuple, Union

from .arith import constant

from ..core.scalar import ScalarTensor
from ..core.meta import MetaTensor
from ..core.tensor import Tensor
from ..core.dispatch import dispatch, register_dispatch
from ..operator.control_flow import ForLoop, WhileLoop, IfBlock, Block
from .utils import _broadcast_constant
from ..trace.function import trace_function
from ..trace.executor import TraceGraphExecutor


def _make_meta_input(arg: Tensor):
    return MetaTensor(tuple(MetaTensor((), dim.dtype) for dim in arg.shape), arg.dtype)


def _make_meta_inputs(*args: Tensor):
    return [_make_meta_input(arg) for arg in args]


def for_loop(count: Tensor, variables: Sequence[Tensor], body: Block) -> Tuple[Tensor, ...]:
    graph = trace_function(body, _make_meta_inputs(count, *variables))
    results = dispatch(ForLoop(graph), count, *variables)
    return results


def while_loop(condition: Tensor, variables: Sequence[Tensor], body: Block) -> Tuple[Tensor, ...]:
    graph = trace_function(body, _make_meta_inputs(condition, *variables))
    results = dispatch(WhileLoop(graph), condition, *variables)
    return results


def if_block(condition: Tensor, variables: Sequence[Tensor], true_body: Block, false_body: Block) -> Tuple[Tensor, ...]:
    true_graph = trace_function(true_body, _make_meta_inputs(*variables))
    false_graph = trace_function(false_body, _make_meta_inputs(*variables))
    results = dispatch(IfBlock(true_graph, false_graph), condition, *variables)
    return results


@register_dispatch()
def dispatch_for_loop(op: ForLoop, count: MetaTensor, *args: Tensor):
    outputs = TraceGraphExecutor(op.body)(count, *args)
    return outputs


@register_dispatch()
def dispatch_for_loop(op: ForLoop, count: ScalarTensor, *args: Tensor):
    body_executor = TraceGraphExecutor(op.body)
    for i in count.item():
        outputs = tuple(body_executor(constant(i, count.dtype), *args))
        args = [*args[-len(outputs):], *outputs]
    return outputs
