from .operator import (
    DistributedSend,
    DistributedRecv,
    DistributedBroadcast,
    DistributedAllGather,
    DistributedReduceScatter,
    DistributedScatter,
    DistributedAllToAll,
    DistributedAllReduce,
)
from ..core.dispatch import dispatch
from ..core.tensor import Tensor


class DistributedCommunicator:
    version: Tensor

    def __init__(self, version: Tensor) -> None:
        self.version = version

    def send(self, x: Tensor, target: int):
        (self.version,) = dispatch(DistributedSend(target), self.version, x)

    def recv(self, source: int):
        (self.version, z) = dispatch(DistributedRecv(source), self.version)
        return z

    def broadcast(self, x: Tensor, source: int):
        (self.version, z) = dispatch(DistributedBroadcast(source), self.version, x)
        return z

    def all_reduce(self, x: Tensor):
        (self.version, z) = dispatch(DistributedAllReduce(), self.version, x)
        return z

    def all_gather(self, x: Tensor, axis: int):
        (self.version, z) = dispatch(DistributedAllGather(axis), self.version, x)
        return z

    def reduce_scatter(self, x: Tensor, axis: int):
        (self.version, z) = dispatch(DistributedReduceScatter(axis), self.version, x)
        return z

    def scatter(self, x: Tensor, source: int, axis: int):
        (self.version, z) = dispatch(DistributedScatter(source, axis), self.version, x)
        return z

    def all_to_all(self, x: Tensor, gather_axis: int, scatter_axis: int):
        (self.version, z) = dispatch(DistributedAllToAll(gather_axis, scatter_axis), self.version, x)
        return z
