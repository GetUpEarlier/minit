from typing import Union

from ..operator.generate import Fill

from ..distributed.group import get_world

from ..core.constant import ConstantTensor

from ..core.tensor import Tensor
from ..core.dispatch import register_dispatch, dispatch
from ..functional.linalg import matrix_multiply
from .spec import CollectiveSpecBroadcast, CollectiveSpecSplit
from ..operator.linalg import MatrixMultiply
from .tensor import CollectiveTensor
from ..core.operator import Operator


def collective_predicate(op, *args):
    return any(arg is CollectiveTensor for arg in args)


@register_dispatch(predicate=collective_predicate, priority=1)
def dispatch_collective(op: Operator, *args: Union[CollectiveTensor, ConstantTensor]):
    communicator = None
    for arg in args:
        if isinstance(arg, CollectiveTensor):
            communicator = arg._communicator
    local_args = [arg.to_broadcast()._local if isinstance(arg, CollectiveTensor) else arg for arg in args]
    for local_arg in local_args:
        assert not isinstance(local_arg, CollectiveTensor)
    local_outputs = dispatch(op, *local_args)
    return tuple([CollectiveTensor.from_broadcast(communicator, local_output) for local_output in local_outputs])
