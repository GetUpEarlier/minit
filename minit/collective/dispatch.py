from minit.cuda import dispatch
from .tensor import CollectiveTensor
from ..core.operator import Operator


def dispatch_collective(op: Operator, *args: CollectiveTensor):
    args = [arg.to_broadcast()._local for arg in args]
    outputs = dispatch(op, *args)
    return tuple([CollectiveTensor.from_broadcast(args[0]._communicator, output) for output in outputs])
