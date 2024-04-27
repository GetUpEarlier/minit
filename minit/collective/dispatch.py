from typing import Literal, Union

from ..core.object import match_pattern
from ..core.tensor import Tensor
from ..core.dispatch import register_dispatch, dispatch
from ..functional.linalg import matrix_multiply
from .spec import CollectiveSpec, CollectiveSpecBroadcast, CollectiveSpecSplit
from ..operator.linalg import MatrixMultiply
from .tensor import CollectiveTensor
from ..core.operator import Operator


@register_dispatch(predicate=lambda *args: any(match_pattern(CollectiveTensor[CollectiveSpec], arg) is not None for arg in args), priority=1)
def dispatch_collective(op: Operator, *args: Union[CollectiveTensor[CollectiveSpec], Tensor]):
    communicator = None
    for arg in args:
        if isinstance(arg, CollectiveTensor):
            communicator = arg._communicator
    local_args = [arg.to_broadcast()._local if isinstance(arg, CollectiveTensor) else arg for arg in args]
    for local_arg in local_args:
        assert not isinstance(local_arg, CollectiveTensor)
    local_outputs = dispatch(op, *local_args)
    return tuple([CollectiveTensor.from_broadcast(communicator, local_output) for local_output in local_outputs])


@register_dispatch(priority=2)
def dispatch_collective(op: MatrixMultiply, x: CollectiveTensor[CollectiveSpec], y: CollectiveTensor[CollectiveSpecSplit[Literal[0]]]):
    communicator = x._communicator
    x = x.to_broadcast()
    return tuple([CollectiveTensor.from_split(communicator, matrix_multiply(x._local, y._local), 1)])


@register_dispatch(priority=2)
def dispatch_collective(op: MatrixMultiply, x: CollectiveTensor[CollectiveSpec], y: CollectiveTensor[CollectiveSpecSplit[Literal[1]]]):
    communicator = x._communicator
    x = x.to_split(1)
    return tuple([CollectiveTensor.from_partial(communicator, matrix_multiply(x._local, y._local))])


@register_dispatch(priority=2)
def dispatch_collective(op: MatrixMultiply, x: CollectiveTensor[CollectiveSpec], y: CollectiveTensor[CollectiveSpecBroadcast]):
    communicator = x._communicator
    x = x.to_broadcast()
    return tuple([CollectiveTensor.from_broadcast(communicator, matrix_multiply(x._local, y._local))])
